import numpy as np
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
import pandas as pd
from tqdm import tqdm
from example_robot_data.robots_loader import load
from multiprocessing import Pool, Manager

def create_single_case(kinDyn, nq, nx, dt, N, qMin, qMax, vMin, vMax, tauMin, tauMax):
    """Creates a single data case with a given configuration."""
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

    lbx = qMin.tolist() + vMin.tolist()
    ubx = qMax.tolist() + vMax.tolist()
    tau_min = tauMin.tolist()
    tau_max = tauMax.tolist()

    X = []
    U = []
    X.append(opti.variable(nx))
    for i in range(1, N + 1):
        X.append(opti.variable(nx))
        opti.subject_to(opti.bounded(lbx, X[-1], ubx))

    for i in range(N):
        U.append(opti.variable(nq))
        opti.subject_to(X[i + 1] == X[i] + dt * f(X[i], U[i]))
        opti.subject_to(opti.bounded(tau_min, inv_dyn(X[i], U[i]), tau_max))
        opti.subject_to(opti.bounded(lbx, X[i], ubx))

    opti.subject_to(X[0] == param_x_init)
    opti.subject_to(X[N - 1] == X[N])

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

    q0 = np.random.uniform(qMin, qMax, nq)
    dq0 = np.random.uniform(vMin, vMax, nq)
    opti.set_value(param_x_init, np.concatenate([q0, dq0]))
    
    try:
        sol = opti.solve()
        return [q0[0], q0[1], dq0[0], dq0[1], 1]
    except:
        return [q0[0], q0[1], dq0[0], dq0[1], 0]

def worker(kinDyn, nq, nx, dt, N, qMin, qMax, vMin, vMax, tauMin, tauMax, num_cases):
    results = []
    for _ in range(num_cases):
        result = create_single_case(kinDyn, nq, nx, dt, N, qMin, qMax, vMin, vMax, tauMin, tauMax)
        results.append(result)
    return results

if __name__ == "__main__":
    time_start = clock()
    print("Load robot model")
    robot = load("double_pendulum")

    print("Create KinDynComputations object")
    joints_name_list = [s for s in robot.model.names[1:]]  # skip the first name because it is "universe"
    nq = len(joints_name_list)  # number of joints
    nx = 2 * nq  # size of the state variable
    kinDyn = KinDynComputations(robot.urdf, joints_name_list)

    dt = 0.01  # time step for optimal control problem
    N = 15  # time horizon

    qMin = np.array([-np.pi, -np.pi])
    qMax = -qMin
    vMax = np.array([8.0, 8.0])
    vMin = -vMax
    tauMax = np.array([1.0, 1.0])
    tauMin = -tauMax

    TOT_NEG = 1000
    TOT_POS = 1000

    df = pd.DataFrame(columns=['Q0_1', 'Q0_2', 'DQ0_1', 'DQ0_2', 'Label'])

    # Parallelization setup
    num_processes = 2  # Adjust based on your system
    num_cases_per_process = (TOT_NEG + TOT_POS) // num_processes

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

    name = 'dataset_N' + str(N) + '.csv'
    df.to_csv(name, index=False)

    print(f"Dataset saved as {name}.")
