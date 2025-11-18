from pydrake.all import RigidTransform
from manipulation.scenarios import AddWsg

def add_robot_and_gripper(plant, parser):
    """
    Adds the robot and gripper to the plant.
    """

    iiwa = parser.AddModelsFromUrl(
        "package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"
    )[0]
    iiwa_base = plant.GetFrameByName("iiwa_link_0", iiwa)

    iiwa_pose = RigidTransform([0, -0.75, 0])   # behind board
    plant.WeldFrames(plant.world_frame(), iiwa_base, iiwa_pose)

    wsg = AddWsg(plant, iiwa, roll=0.0, welded=True)
    return plant, iiwa, wsg
