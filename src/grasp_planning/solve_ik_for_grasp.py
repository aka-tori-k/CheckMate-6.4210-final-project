from pydrake.all import (
    InverseKinematics, Solve, RotationMatrix
)
import numpy as np

def solve_ik_for_grasp(plant, plant_context, wsg, T_world_grasp, q_seed=None):
    """
    Solve for iiwa joint positions that place the gripper at the desired grasp pose.
    T_world_grasp: RigidTransform
    """
    if q_seed is None:
        q_seed = plant.GetPositions(plant_context)

    gripper_frame = plant.GetFrameByName("body", wsg)

    ik = InverseKinematics(plant, plant_context)

    ## STRICT CONSTRAINTS ##
    # # Position constraint: place gripper frame origin at desired translation
    # ik.AddPositionConstraint(
    #     gripper_frame,
    #     [0, 0, 0],
    #     plant.world_frame(),
    #     T_world_grasp.translation(),
    #     T_world_grasp.translation(),
    # )

    # # Orientation constraint: align gripper frame with desired orientation
    # ik.AddOrientationConstraint(
    #     gripper_frame,
    #     RotationMatrix(),
    #     plant.world_frame(),
    #     T_world_grasp.rotation(),
    #     0.0,   # angle tolerance
    # )

    ## LOOSE CONSTRAINTS ##
    # Position constraint with tolerance (1 cm box)
    pos_tol = 0.005
    p_WG = T_world_grasp.translation()

    ik.AddPositionConstraint(
        frameB=gripper_frame,
        p_BQ=[0, 0, 0],
        frameA=plant.world_frame(),
        p_AQ_lower=p_WG - pos_tol,
        p_AQ_upper=p_WG + pos_tol,
    )

    # Orientation constraint with tolerance (5 degrees)
    theta_tol = np.deg2rad(3)

    ik.AddOrientationConstraint(
        gripper_frame,
        RotationMatrix(),
        plant.world_frame(),
        T_world_grasp.rotation(),
        theta_tol,
    )

    prog = ik.get_mutable_prog()
    q = ik.q()

    # Optional: cost to stay close to seed
    prog.AddQuadraticErrorCost(np.eye(len(q)), q_seed, q)
    prog.SetInitialGuess(q, q_seed)

    result = Solve(prog)
    if not result.is_success():
        # print("IK failed for this grasp")
        return None

    q_sol = result.GetSolution(q)
    return q_sol[:7]


