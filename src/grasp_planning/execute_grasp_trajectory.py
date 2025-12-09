from pydrake.all import PiecewisePolynomial

CLOSE = 0.002
OPEN = 0.107

def make_gripper_traj(start, end, duration=0.5):
    return PiecewisePolynomial.FirstOrderHold(
        [0.0, duration],
        [[start, end]]
    )

def execute_grasp_trajectory(traj, simulator, wsg_traj_source):
    print("Uploading WSG trajectory...")
    wsg_traj_source.set_trajectory(traj)

    ctx = simulator.get_context()
    t_now = ctx.get_time()
    t_end = traj.end_time() + 0.05

    print(f"Simulating gripper from t={t_now:.3f} to t={t_end:.3f}")
    simulator.AdvanceTo(t_end)

if __name__ == "__main__":

    # print("Closing gripper…")
    # traj = make_gripper_traj(0.107, 0.002)
    # shifted_traj = shift_trajectory(traj, simulator.get_context().get_time())
    # execute_grasp_trajectory(shifted_traj, simulator, wsg_traj_source)
    # print("gripper is (hypothetically) closed")
    
    while True:
        pass