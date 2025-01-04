import numpy as np
from adam.casadi.computations import KinDynComputations
import casadi as cs
import pandas as pd
from example_robot_data.robots_loader import load
from multiprocessing import Pool, Manager
from time import time as clock
import os

def create_single_case(kinDyn, nq, nx, dt, N, qMin, qMax, vMin, vMax, tauMin, tauMax, positive_left, negative_left):
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
    
    # Create the optimization problem
    opti = cs.Opti()
    param_x_init = opti.parameter(nx)

    # Create the dynamics function
    q = cs.SX.sym('q', nq)
    dq = cs.SX.sym('dq', nq)
    ddq = cs.SX.sym('ddq', nq)
    state = cs.vertcat(q, dq)
    rhs = cs.vertcat(dq, ddq)
    f = cs.Function('f', [state, ddq], [rhs])

    H_b = cs.SX.eye(4)  # base configuration
    v_b = cs.SX.zeros(6)  # base velocity
    bias_forces = kinDyn.bias_force_fun()
    mass_matrix = kinDyn.mass_matrix_fun()

    h = bias_forces(H_b, q, v_b, dq)[6:]
    M = mass_matrix(H_b, q)[6:, 6:]
    tau = M @ ddq + h

    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

    
    # Get the bounds
    lbx = qMin.tolist() + vMin.tolist()
    ubx = qMax.tolist() + vMax.tolist()
    tau_min = tauMin.tolist()
    tau_max = tauMax.tolist()

    
    # Create the optimization variables
    X = []
    U = []
    X.append(opti.variable(nx))
    for i in range(1, N + 1):
        X.append(opti.variable(nx))
        # bound the states
        opti.subject_to(opti.bounded(lbx, X[-1], ubx))

    for i in range(N):
        U.append(opti.variable(nq))
        # dynamics constraint
        opti.subject_to(X[i + 1] == X[i] + dt * f(X[i], U[i]))
        # bound the torques
        opti.subject_to(opti.bounded(tau_min, inv_dyn(X[i], U[i]), tau_max))
    
    # initial constraint
    opti.subject_to(X[0] == param_x_init)
    # final constraint used for checking if the state is in N-step backward reachable set of S
    opti.subject_to(X[N - 1] == X[N])
    # set the cost to 1
    cost = 1
    opti.minimize(cost)

    opts = {
        "ipopt.print_level": 0,
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-6,
        "ipopt.compl_inf_tol": 1e-6,
        "print_time": 0,
        "detect_simple_bounds": True,
        "ipopt.max_iter": 1000
    }
    opti.solver("ipopt", opts)

    # Randomly sample initial state
    q0 = np.random.uniform(qMin, qMax, nq)
    dq0 = np.random.uniform(vMin, vMax, nq)
    # set it as the initial value
    opti.set_value(param_x_init, np.concatenate([q0, dq0]))
    
    try:
        opti.solve()
        if positive_left > 0:
            return [q0[0], q0[1], dq0[0], dq0[1], 1]    # 1 means the state is in the N-step backward reachable set of S
        else:
            return False    
    except:
        if negative_left > 0:
            return [q0[0], q0[1], dq0[0], dq0[1], 0]    # 0 means the state is not in the N-step backward reachable set of S
        else:
            return False

def worker(kinDyn, nq, nx, dt, N, qMin, qMax, vMin, vMax, tauMin, tauMax, num_cases):
    """
    Worker function to generate multiple data cases in parallel.

    Parameters:
        kinDyn, nq, nx, dt, N, qMin, qMax, vMin, vMax, tauMin, tauMax: As described above.
        num_cases (int): Number of data cases to generate.

    Returns:
        list: List of generated cases.
    """
    results = []
    positive_left = num_cases // 2
    negative_left = num_cases // 2

    # Set a unique random seed based on the process ID
    np.random.seed(os.getpid())

    while positive_left > 0 or negative_left > 0:
        result = create_single_case(kinDyn, nq, nx, dt, N, qMin, qMax, vMin, vMax, tauMin, tauMax, positive_left, negative_left)
        if result:
            if result[-1] == 1:
                positive_left -= 1
                if positive_left % 10 == 0:
                    print(f"pos: {positive_left}")
            else:
                negative_left -= 1
                if negative_left % 10 == 0:
                    print(f"neg: {negative_left}")
            results.append(result)
    return results

if __name__ == "__main__":
    time_start = clock()
    print("Load robot model")
    robot = load("double_pendulum")
    print("Create KinDynComputations object")
    joints_name_list = [s for s in robot.model.names[1:]]
    nq = len(joints_name_list)  # number of joints
    nx = 2 * nq  # size of the state variable
    kinDyn = KinDynComputations(robot.urdf, joints_name_list)

    dt = 0.01  # time step for optimal control problem
    N = 15  # 15-step backward reachable set of S

    # Define the bounds
    qMin = np.array([-np.pi, -np.pi])
    qMax = -qMin
    vMax = np.array([8.0, 8.0])
    vMin = -vMax
    tauMax = np.array([1.0, 1.0])
    tauMin = -tauMax

    # Define the number of positive and negative cases
    TOT_NEG = 1000
    TOT_POS = 1000

    # Create the DataFrame
    df = pd.DataFrame(columns=['Q0_1', 'Q0_2', 'DQ0_1', 'DQ0_2', 'Label'])

    # Parallelization setup
    num_processes = 2  # number of processes
    num_cases_per_process = (TOT_NEG + TOT_POS) // num_processes

    print(f"Creating dataset with {TOT_NEG} negative and {TOT_POS} positive cases.")
    
    print("Start creating dataset...")
    with Pool(processes=num_processes) as pool:
        # Distribute workload
        tasks = [
            (kinDyn, nq, nx, dt, N, qMin, qMax, vMin, vMax, tauMin, tauMax, num_cases_per_process)
            for _ in range(num_processes)
        ]

        results = pool.starmap(worker, tasks)

    # Flatten results and create the DataFrame
    for res in results:
        df = pd.concat([df, pd.DataFrame(res, columns=['Q0_1', 'Q0_2', 'DQ0_1', 'DQ0_2', 'Label'])])
    print("Dataset created.")
    name = 'dataset_N' + str(N) + '.csv'
    df.to_csv(name, index=False)
    print(f"Dataset saved as {name}. Time elapsed: {clock() - time_start:.2f} seconds.")

