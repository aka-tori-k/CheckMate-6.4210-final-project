import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")
from path_planning.rrt_connect import IiwaProblem

def generate_iiwa_problem(q_start, q_goal, plant, scene_graph, diagram, diagram_context, plant_context):
    problem = IiwaProblem(
        q_start=q_start,
        q_goal=q_goal,
        gripper_setpoint=0.02,
        is_visualizing=False,
        plant=plant,
        scene_graph=scene_graph,
        diagram=diagram,
        diagram_context=diagram_context,
        plant_context=plant_context,
    )

    def is_collision_free(q):
        return not problem.collide(q)  # True if q is safe
    return problem, is_collision_free