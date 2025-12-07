import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from simulation_setup.initialize_simulation import initialize_simulation
import time
from random import random
from typing import Literal

import numpy as np
from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    Rgba,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    SolutionResult,
    Solve,
    StartMeshcat,
    TrajectorySource,
)
from pydrake.multibody import inverse_kinematics
from pydrake.trajectories import PiecewisePolynomial

from manipulation.exercises.trajectories.rrt_planner.robot import (
    ConfigurationSpace,
    Range,
)
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import (
    RRT,
    Problem,
    TreeNode,
)
from manipulation.meshcat_utils import AddMeshcatTriad


class IiwaProblem(Problem):
    def __init__(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        gripper_setpoint: float,
        is_visualizing=False,
        plant: MultibodyPlant = None,
        scene_graph: any = None,
        diagram_context: any = None,
        plant_context: any = None,
    ) -> None:
        self.gripper_setpoint = gripper_setpoint
        self.is_visualizing = is_visualizing
        self.plant = plant
        self.scene_graph = scene_graph
        self.diagram_context = diagram_context
        self.plant_context = plant_context
        self.sg_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)
        self.query_port = scene_graph.get_query_output_port()

        # --- Check start configuration ---
        self.start_in_collision = self.collide(q_start)

        # --- Check goal configuration ---
        self.goal_in_collision = self.collide(q_goal)

        
        # Construct configuration space for IIWA.
        nq = 7
        joint_limits = np.zeros((nq, 2))
        for i in range(nq):
            joint = plant.GetJointByName("iiwa_joint_%i" % (i + 1))
            joint_limits[i, 0] = joint.position_lower_limits().item()
            joint_limits[i, 1] = joint.position_upper_limits().item()
        
        self.joint_limits = joint_limits.copy()

        range_list = []
        for joint_limit in joint_limits:
            range_list.append(Range(joint_limit[0], joint_limit[1]))

        def l2_distance(q: tuple):
            sum = 0
            for q_i in q:
                sum += q_i**2
            return np.sqrt(sum)

        max_steps = nq * [np.pi / 180]  # two degrees
        cspace_iiwa = ConfigurationSpace(range_list, l2_distance, max_steps)

        # Call base class constructor.
        Problem.__init__(
            self,
            x=10,  # not used.
            y=10,  # not used.
            robot=None,  # not used.
            obstacles=None,  # not used.
            start=tuple(q_start),
            goal=tuple(q_goal),
            cspace=cspace_iiwa,
        )

    def collide(self, q: np.ndarray) -> bool:
        plant = self.plant
        plant_context = self.plant_context
        iiwa = plant.GetModelInstanceByName("iiwa7")

        # Set positions
        plant.SetPositions(plant_context, iiwa, q)

        # Reuse cached context + query port
        query_object = self.query_port.Eval(self.sg_context)

        return query_object.HasCollisions()



    def valid_configuration(self, configuration: np.ndarray) -> bool:
        q = np.array(configuration)

        # Joint limit check using our stored limits
        for i, qi in enumerate(q):
            lower = self.joint_limits[i, 0]
            upper = self.joint_limits[i, 1]
            if not (lower <= qi <= upper):
                return False

        # No collision checking for now
        return True

    def visualize_iiwa_movement(self, path, plant, diagram, diagram_context, plant_context, rate=20):
        """
        Visualizes the RRT path using the MultibodyPlant + Diagram from initialize_simulation().
        """
        import time

        # plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

        dt = 1.0 / rate

        iiwa_model = plant.GetModelInstanceByName("iiwa7")

        for q in path:
            q = np.array(q)

            # Set joint positions
            plant.SetPositions(plant_context, iiwa_model, q)

            # Publish to Meshcat through the diagram
            diagram.ForcedPublish(diagram_context)

            time.sleep(dt)

    def draw_path(self, path, plant, diagram_context, plant_context, meshcat, link_name="iiwa_link_7"):
        frame_EE = plant.GetFrameByName(link_name)

        pts = []

        # create a temporary context so we don't overwrite the real robot position
        tmp_context = plant.CreateDefaultContext()

        for q in path:
            q = np.array(q)

            # set q in temp context ONLY
            plant.SetPositions(tmp_context, plant.GetModelInstanceByName("iiwa7"), q)

            # compute EE pose w.r.t. temp context
            X_WE = frame_EE.CalcPoseInWorld(tmp_context)
            pts.append(X_WE.translation())

        pts = np.vstack(pts)

        meshcat.SetLine(
            path="/rrt_path",
            vertices=np.asfortranarray(pts.T),
            line_width=0.01,
            rgba=Rgba(1, 0, 0, 1)
        )

class RRT_tools:
    def __init__(self, problem: IiwaProblem) -> None:
        # rrt is a tree
        self.rrt_tree = RRT(TreeNode(problem.start), problem.cspace)
        problem.rrts = [self.rrt_tree]
        self.problem = problem

    def find_nearest_node_in_RRT_graph(self, q_sample: tuple) -> TreeNode:
        nearest_node = self.rrt_tree.nearest(q_sample)
        return nearest_node

    def sample_node_in_configuration_space(self) -> tuple:
        q_sample = self.problem.cspace.sample()
        return q_sample

    def calc_intermediate_qs_wo_collision(
        self, q_start: tuple, q_end: tuple
    ) -> list[tuple]:
        """create more samples by linear interpolation from q_start
        to q_end. Return all samples that are not in collision

        Example interpolated path:
        q_start, qa, qb, (Obstacle), qc , q_end
        returns >>> q_start, qa, qb
        """
        return self.problem.safe_path(q_start, q_end)

    def grow_rrt_tree(self, parent_node: TreeNode, q_sample: tuple) -> TreeNode:
        """
        add q_sample to the rrt tree as a child of the parent node
        returns the rrt tree node generated from q_sample
        """
        child_node = self.rrt_tree.add_configuration(parent_node, q_sample)
        return child_node

    def node_reaches_goal(self, node: TreeNode, tol: float = 1e-2) -> bool:
        "returns true if the node is within tol of goal, false otherwise"
        return self.problem.cspace.distance(node.value, self.problem.goal) <= tol

    def backup_path_from_node(self, node: TreeNode) -> list[tuple]:
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path
    


def rrt_planning(
    problem: IiwaProblem, max_iterations: int = 1000, prob_sample_q_goal: float = 0.05
) -> tuple[list[tuple] | None, int]:

    rrt_tools = RRT_tools(problem)
    q_goal = problem.goal
    q_start = problem.start

    for k in range(max_iterations):

        # --- 1. Sample a configuration (goal-biased sampling) ---
        r = random()
        if r < prob_sample_q_goal:
            q_sample = q_goal
        else:
            q_sample = rrt_tools.sample_node_in_configuration_space()

        # --- 2. Find nearest existing tree node ---
        nearest_node = rrt_tools.find_nearest_node_in_RRT_graph(q_sample)

        # --- 3. Compute intermediate collision-free nodes toward sample ---
        safe_path = rrt_tools.calc_intermediate_qs_wo_collision(
            nearest_node.value, q_sample
        )

        if len(safe_path) == 0:
            # Entire path collided at first step
            continue

        # safe_path includes q_start… last-valid… but we only add last-valid
        q_new = safe_path[-1]

        # If the nearest node is exactly q_new, no extension happened
        if np.allclose(q_new, nearest_node.value):
            continue

        # --- 4. Add new configuration to RRT graph ---
        new_node = rrt_tools.grow_rrt_tree(nearest_node, q_new)

        # --- 5. Check if we reached the goal ---
        if rrt_tools.node_reaches_goal(new_node):
            # Build/return full path from start to goal
            path = rrt_tools.backup_path_from_node(new_node)
            return path, k + 1

    # Failed: return no solution
    return None, max_iterations
    

    
class RRT_Connect_tools(RRT_tools):
    def create_new_tree(self, q_root: tuple[float]) -> RRT:
        return RRT(TreeNode(q_root), self.problem.cspace)

    def extend_once(
        self, tree: RRT, q_target: tuple[float], eps_connect: float = 1e-3
    ) -> tuple[Literal["Trapped", "Reached", "Advanced"], TreeNode]:
        "extends tree by one step towards q_target"
        q_near_node = tree.nearest(q_target)
        edge = self.problem.safe_path(q_near_node.value, q_target)
        if len(edge) <= 1:
            return "Trapped", q_near_node

        q_step = edge[1]
        new_node = tree.add_configuration(q_near_node, q_step)

        reached = q_step == q_target
        if not reached:
            if self.problem.cspace.distance(q_step, q_target) <= eps_connect:
                tail_edge = self.problem.safe_path(q_step, q_target)
                if len(tail_edge) > 1 and tail_edge[-1] == q_target:
                    new_node = tree.add_configuration(new_node, q_target)
                    reached = True

        return ("Reached" if reached else "Advanced"), new_node

    def connect_greedy(
        self, tree: RRT, q_target: tuple[float], eps_connect: float = 1e-3
    ) -> tuple[Literal["Trapped", "Reached", "Advanced"], TreeNode | None]:
        status, last = "Advanced", None
        while status == "Advanced":
            status, last = self.extend_once(tree, q_target, eps_connect)
            if status == "Trapped":
                return "Trapped", last
        return "Reached", last

    @staticmethod
    def concat_paths(path_a: list[tuple], path_b: list[tuple]) -> list[tuple]:
        if path_a and path_b and path_a[-1] == path_b[0]:
            return path_a + path_b[1:]
        return path_a + path_b
    
def rrt_connect_planning(
    problem: IiwaProblem, max_iterations: int = 1000, eps_connect: float = 1e-2
) -> tuple[list[tuple] | None, int]:
    
    tools = RRT_Connect_tools(problem)
    T_start = tools.rrt_tree                  # Tree rooted at q_start
    T_goal  = tools.create_new_tree(problem.goal)  # Tree rooted at q_goal

    for k in range(max_iterations):

        # -------- 1. Sample ----------
        q_rand = tools.sample_node_in_configuration_space()

        # -------- 2. Extend T_start toward sample ----------
        status, new_node = tools.extend_once(T_start, q_rand, eps_connect)
        if status == "Trapped":
            # Nothing happened this iteration; continue
            continue

        # -------- 3. Try to connect T_goal to new_node ----------
        connect_status, goal_last = tools.connect_greedy(
            T_goal, new_node.value, eps_connect
        )

        if connect_status == "Reached":
            # --- FOUND SOLUTION ---
            # Extract paths: start → new_node and goal → meeting point
            path_start = tools.backup_path_from_node(new_node)
            path_goal  = tools.backup_path_from_node(goal_last)
            path_goal.reverse()  # Because T_goal is rooted at q_goal

            full_path = tools.concat_paths(path_start, path_goal)
            return full_path, k + 1

        # -------- 4. Swap the start and goal trees ----------
        T_start, T_goal = T_goal, T_start

    # Failed to find a path within max_iterations
    return None, max_iterations


def sample_random_q(plant):
    q = []
    for i in range(7):
        joint = plant.GetJointByName(f"iiwa_joint_{i+1}")
        lo = joint.position_lower_limits()[0]
        hi = joint.position_upper_limits()[0]
        q.append(np.random.uniform(lo, hi))
    return np.array(q)

if __name__ == "__main__":
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram = initialize_simulation()
    q_start = sample_random_q(plant)
    q_goal  = sample_random_q(plant)

    print("Random start:", q_start)
    print("Random goal:", q_goal)
    
    problem = IiwaProblem(
        q_start=q_start,
        q_goal=q_goal,
        gripper_setpoint=0.02,
        is_visualizing=True,
        plant=plant,
        scene_graph=scene_graph,
        diagram_context=diagram_context,
        plant_context=plant_context,
    )
    path, iters = rrt_connect_planning(problem, max_iterations=5000, eps_connect=0.05)
    # print(path)
    print(f"RRT-Connect finished in {iters} iterations.")
    if path is not None:
        print(f"Found path with {len(path)} waypoints.")
        problem.draw_path(path, plant, diagram_context, plant_context, meshcat, link_name="iiwa_link_7")
        problem.visualize_iiwa_movement(path, plant, diagram, diagram_context, plant_context)
        while True:
            pass
    else:
        print("No path found.")
    
    
