import numpy as np
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
import pandas as pd
from tqdm import tqdm
from example_robot_data.robots_loader import load
from multiprocessing import Pool, Manager

def create_single_case(kinDyn, nq, nx, dt, N, lbx, ubx, tau_min, tau_max, effort_limit, velocity_limit, position_limit):
    """
    Creates a single data case for the UR5 robot.
    
    Parameters:
        kinDyn (KinDynComputations): The kinematic and dynamic computations object.
        nq (int): Number of joints.
        nx (int): Size of the state variable.
        dt (float): Time step for the optimization problem.
        N (int): Number of steps in the backward reachable set.
        lbx (list): Lower bounds for state variables.
        ubx (list): Upper bounds for state variables.
        tau_min (list): Minimum torque limits.
        tau_max (list): Maximum torque limits.
        effort_limit (np.array): Effort (torque) limits for the robot joints.
        velocity_limit (np.array): Velocity limits for the robot joints.
        position_limit (list): Position limits [lower, upper] for the robot joints.
    
    Returns:
        list: A single case containing initial states and a label indicating reachability.
    """
    opti = cs.Opti()
    param_x_init = opti.parameter(nx)

    # Define symbolic variables for state and dynamics
    q = cs.SX.sym('q', nq)
    dq = cs.SX.sym('dq', nq)
    ddq = cs.SX.sym('ddq', nq)
    state = cs.vertcat(q, dq)
    rhs = cs.vertcat(dq, ddq)
    f = cs.Function('f', [state, ddq], [rhs])

    # Kinematic and dynamic calculations
    H_b = cs.SX.eye(4)  # Base configuration
    v_b = cs.SX.zeros(6)  # Base velocity
    bias_forces = kinDyn.bias_force_fun()
    mass_matrix = kinDyn.mass_matrix_fun()

    h = bias_forces(H_b, q, v_b, dq)[6:]
    M = mass_matrix(H_b, q)[6:, 6:]
    tau = M @ ddq + h

    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

    # Create optimization variables
    X = [opti.variable(nx) for _ in range(N + 1)]
    U = [opti.variable(nq) for _ in range(N)]

    # Dynamics and bounds constraints
    for i in range(N):
        opti.subject_to(X[i + 1] == X[i] + dt * f(X[i], U[i]))
        opti.subject_to(opti.bounded(lbx, X[i], ubx))
        opti.subject_to(opti.bounded(tau_min, inv_dyn(X[i], U[i]), tau_max))

    # Initial and final constraints
    opti.subject_to(X[0] == param_x_init)
    # final constraint used for checking if the state is in N-step backward reachable set of S
    opti.subject_to(X[N - 1] == X[N])

    # Minimize constant cost
    cost = 1
    opti.minimize(cost)

    # Solver options
    opts = {
        "ipopt.print_level": 0,
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-6,
        "ipopt.compl_inf_tol": 1e-6,
        "print_time": 0,
        "detect_simple_bounds": True
    }
    opti.solver("ipopt", opts)

    # Random sampling of initial state
    q0 = np.random.uniform(position_limit[0], position_limit[1], nq)
    dq0 = np.random.uniform(-velocity_limit, velocity_limit, nq)
    opti.set_value(param_x_init, np.concatenate([q0, dq0]))

    try:
        opti.solve()
        return [*q0, *dq0, 1]  # State is in the N-step backward reachable set
    except:
        return [*q0, *dq0, 0]  # State is not in the N-step backward reachable set

def worker(kinDyn, nq, nx, dt, N, lbx, ubx, tau_min, tau_max, effort_limit, velocity_limit, position_limit, num_cases):
    """
    Worker function to generate multiple data cases in parallel.

    Parameters:
        kinDyn, nq, nx, dt, N, lbx, ubx, tau_min, tau_max, effort_limit, velocity_limit, position_limit: As described above.
        num_cases (int): Number of data cases to generate.

    Returns:
        list: List of generated cases.
    """
    results = []
    for _ in range(num_cases):
        result = create_single_case(kinDyn, nq, nx, dt, N, lbx, ubx, tau_min, tau_max, effort_limit, velocity_limit, position_limit)
        results.append(result)
    return results

if __name__ == "__main__":
    time_start = clock()
    print("Load robot model")
    robot = load("ur5")

    print("Create KinDynComputations object")
    joints_name_list = [s for s in robot.model.names[1:]]
    nq = len(joints_name_list)  # Number of joints
    nx = 2 * nq  # Size of the state variable
    kinDyn = KinDynComputations(robot.urdf, joints_name_list)

    dt = 0.01  # Time step
    N = 15  # Number of steps in backward reachable set

    # Define bounds
    position_limit = [robot.model.lowerPositionLimit, robot.model.upperPositionLimit]
    velocity_limit = robot.model.velocityLimit
    effort_limit = robot.model.effortLimit * 0.5

    lbx = position_limit[0].tolist() + (-velocity_limit).tolist()
    ubx = position_limit[1].tolist() + velocity_limit.tolist()
    tau_min = (-effort_limit).tolist()
    tau_max = effort_limit.tolist()

    # Number of positive and negative cases
    TOT_NEG = 1000
    TOT_POS = 1000

    # Initialize the DataFrame
    df = pd.DataFrame(columns=[f'Q0_{i + 1}' for i in range(nq)] + [f'DQ0_{i + 1}' for i in range(nq)] + ['Label'])

    num_processes = 2  # Number of parallel processes
    num_cases_per_process = (TOT_NEG + TOT_POS) // num_processes

    print("Start creating dataset...")
    with Pool(processes=num_processes) as pool:
        # Distribute workload
        tasks = [
            (kinDyn, nq, nx, dt, N, lbx, ubx, tau_min, tau_max, effort_limit, velocity_limit, position_limit, num_cases_per_process)
            for _ in range(num_processes)
        ]
        results = pool.starmap(worker, tasks)

    # Combine results and save to DataFrame
    for res in results:
        df = pd.concat([df, pd.DataFrame(res, columns=df.columns)])

    name = 'dataset_N' + str(N) + '_UR5.csv'
    df.to_csv(name, index=False)

    print(f"Dataset saved as {name}. Time elapsed: {clock() - time_start:.2f} seconds.")
