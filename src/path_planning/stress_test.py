import time
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/workspaces/CheckMate-6.4210-final-project/src")

from path_planning.rrt_connect import (
    IiwaProblem,
    rrt_connect_planning,
    sample_random_q,
)

from simulation_setup.initialize_simulation import initialize_simulation

def sample_valid_q(problem, max_attempts=20):
    for _ in range(max_attempts):
        q = sample_random_q(problem.plant)
        if not problem.collide(q):
            return q
    return None

def set_iiwa_problem(q_start, q_goal, plant, scene_graph, diagram_context, plant_context):
    problem = IiwaProblem(
        q_start=q_start,
        q_goal=q_goal,
        gripper_setpoint=0.02,
        is_visualizing=False,
        plant=plant,
        scene_graph=scene_graph,
        diagram_context=diagram_context,
        plant_context=plant_context,
    )
    return problem

def stress_test(num_trials=20, eps_connect=0.05, max_iterations=5000):
    simulator, plant, plant_context, meshcat, scene_graph, diagram_context, meshcat, diagram = initialize_simulation()

    results = []
    print("\n========== STARTING STRESS TEST ==========\n")

    for trial in range(num_trials):

        print(f"\n================ Trial {trial+1}/{num_trials} ================")

        q_start = sample_random_q(plant)
        q_goal  = sample_random_q(plant)

        problem = set_iiwa_problem(q_start, q_goal, plant, scene_graph, diagram_context, plant_context)

        # -------------------------------------------------
        #     CHECK START & GOAL COLLISION BEFORE PLANNING
        # -------------------------------------------------
        if problem.start_in_collision:
            new_q_start = sample_valid_q(problem)
            if new_q_start is None:
                print("→ Skipping: START is in collision")
                results.append({
                    "success": False,
                    "reason": "start_collision",
                    "q_start": q_start,
                    "q_goal": q_goal,
                    "time": 0,
                    "iterations": 0,
                    "path_length": -1,
                    "path": None,
                })
                continue
            else:
                print("→ Resampled START to avoid collision")
                problem = set_iiwa_problem(new_q_start, q_goal, plant, scene_graph, diagram_context, plant_context)

        if problem.goal_in_collision:
            new_q_goal = sample_valid_q(problem)
            if new_q_goal is None:
                print("→ Skipping: GOAL is in collision")
                results.append({
                    "success": False,
                    "reason": "goal_collision",
                    "q_start": q_start,
                    "q_goal": q_goal,
                    "time": 0,
                    "iterations": 0,
                    "path_length": -1,
                    "path": None,
                })
            else:
                print("→ Resampled GOAL to avoid collision")
                problem = set_iiwa_problem(q_start, new_q_goal, plant, scene_graph, diagram_context, plant_context)

        # -----------------------------
        #     RUN RRT PLANNING
        # -----------------------------
        t0 = time.time()
        path = None
        iters = max_iterations
        
        try:
            path, iters = rrt_connect_planning(problem,
                                               max_iterations=max_iterations,
                                               eps_connect=eps_connect)
        except Exception as e:
            print(f"→ Planner ERROR: {e}")
            results.append({
                "success": False,
                "reason": "exception_during_rrt",
                "q_start": q_start,
                "q_goal": q_goal,
                "time": 0,
                "iterations": iters,
                "path_length": -1,
                "path": None,
            })
            continue

        t1 = time.time()
        duration = t1 - t0

        # success if path exists
        success = path is not None
        path_length = len(path) if success else -1

        if success:
            print(f"→ SUCCESS in {duration:.3f}s, {iters} iterations, {path_length} waypoints")
            results.append({
                "success": True,
                "reason": "success",
                "q_start": q_start,
                "q_goal": q_goal,
                "time": duration,
                "iterations": iters,
                "path_length": path_length,
                "path": path,
            })
        else:
            print("→ FAILED: No path found")
            results.append({
                "success": False,
                "reason": "rrt_failed",
                "q_start": q_start,
                "q_goal": q_goal,
                "time": duration,
                "iterations": iters,
                "path_length": -1,
                "path": None,
            })

    return results


# ----------------------------------------------------------
#                 PLOTTING FUNCTIONS
# ----------------------------------------------------------

def plot_stats(results):
    successes = [r for r in results if r["success"]]
    failures  = [r for r in results if not r["success"]]

    # -------------------------------------------
    # PLOT 1: Scatter (path length vs time)
    # -------------------------------------------
    if successes:
        times = [r["time"] for r in successes]
        lengths = [r["path_length"] for r in successes]
        plt.figure()
        plt.scatter(lengths, times)
        plt.xlabel("Path length (# waypoints)")
        plt.ylabel("Planning time (s)")
        plt.title("RRT-Connect Planning Time vs Path Length")
        plt.grid()
        plt.show()

    # -------------------------------------------
    # PLOT 2: Histogram of planning time
    # -------------------------------------------
    if successes:
        plt.figure()
        plt.hist([r["time"] for r in successes], bins=20)
        plt.xlabel("Planning time (s)")
        plt.ylabel("Count")
        plt.title("Distribution of RRT Planning Times (Successes Only)")
        plt.grid()
        plt.show()

    # -------------------------------------------
    # PLOT 3: Histogram of path lengths
    # -------------------------------------------
    if successes:
        plt.figure()
        plt.hist([r["path_length"] for r in successes], bins=20)
        plt.xlabel("Path length (# waypoints)")
        plt.ylabel("Count")
        plt.title("Distribution of RRT Path Lengths (Successes Only)")
        plt.grid()
        plt.show()

    # -------------------------------------------
    # PLOT 4: Failure reasons bar chart
    # -------------------------------------------
    if failures:
        reasons = [r["reason"] for r in failures]
        uniq = sorted(set(reasons))
        counts = [reasons.count(u) for u in uniq]

        plt.figure()
        plt.bar(uniq, counts)
        plt.xlabel("Failure reason")
        plt.ylabel("Count")
        plt.title("Failure Modes Across Trials")
        plt.xticks(rotation=30)
        plt.grid(axis='y')
        plt.show()



# ----------------------------------------------------------
#                   MAIN ENTRY POINT
# ----------------------------------------------------------

if __name__ == "__main__":
    results = stress_test()

    successes = [r for r in results if r["success"]]
    failures  = [r for r in results if not r["success"]]

    print("\n===================================")
    print("          STRESS TEST SUMMARY     ")
    print("===================================")
    print(f"Total trials: {len(results)}")
    print(f"Successes:    {len(successes)}")
    print(f"Failures:     {len(failures)}")
    print(f"Success rate: {100*len(successes)/len(results):.1f}%")

    if successes:
        avg_time = np.mean([r["time"] for r in successes])
        avg_len  = np.mean([r["path_length"] for r in successes])
        print(f"Avg planning time: {avg_time:.3f}s")
        print(f"Avg path length:  {avg_len:.1f} waypoints")

        best = min(successes, key=lambda r: r["time"])
        worst = max(successes, key=lambda r: r["time"])
        print(f"Fastest time: {best['time']:.3f}s")
        print(f"Slowest time: {worst['time']:.3f}s")

    # Make plots
    plot_stats(results)
