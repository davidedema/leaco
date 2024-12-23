import numpy as np
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock

import pandas as pd
from tqdm import tqdm
from example_robot_data.robots_loader import load

time_start = clock()
print("Load robot model")
robot = load("ur5")

print("Create KinDynComputations object")
joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"
nq = len(joints_name_list)  # number of joints
nx = 2*nq # size of the state variable
kinDyn = KinDynComputations(robot.urdf, joints_name_list)

dt = 0.01 # time step for optimal control problem
N = 3  # time horizon
q0 = np.zeros(nq)  # initial joint configuration
dq0= np.zeros(nq)  # initial joint velocities
done = False
num_negative = 0
num_positive = 0

TOT_NEG = 300
TOT_POS = 200

df = pd.DataFrame(columns=['Q0_1', 'Q0_2', 'Q0_3', 'Q0_4', 'Q0_5', 'Q0_6','DQ0_1', 'DQ0_2', 'DQ0_3', 'DQ0_4', 'DQ0_5', 'DQ0_6', 'Label'])

# Initialize the tqdm progress bar
progress_bar = tqdm(total=TOT_POS + TOT_NEG, desc="Progress", unit="cases")

done = False
while not done:
    opti = cs.Opti()
    param_x_init = opti.parameter(nx)

    # create the dynamics function
    q   = cs.SX.sym('q', nq)
    dq  = cs.SX.sym('dq', nq)
    ddq = cs.SX.sym('ddq', nq)
    state = cs.vertcat(q, dq)
    rhs    = cs.vertcat(dq, ddq)
    f = cs.Function('f', [state, ddq], [rhs])

    # create a Casadi inverse dynamics function
    H_b = cs.SX.eye(4)     # base configuration
    v_b = cs.SX.zeros(6)   # base velocity
    bias_forces = kinDyn.bias_force_fun()
    mass_matrix = kinDyn.mass_matrix_fun()
    # discard the first 6 elements because they are associated to the robot base
    h = bias_forces(H_b, q, v_b, dq)[6:]
    M = mass_matrix(H_b, q)[6:,6:]   # remove degrees of freedom associated to the base
    tau = M @ ddq + h

    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

    # pre-compute state [x] and torque bounds
    lbx = robot.model.lowerPositionLimit.tolist() + (-robot.model.velocityLimit).tolist()
    ubx = robot.model.upperPositionLimit.tolist() + robot.model.velocityLimit.tolist()
    tau_min = (-robot.model.effortLimit).tolist()
    tau_max = robot.model.effortLimit.tolist()

    # create the decision variables, the cost and the constraint

    X = []
    U = []
    for i in range(N+1):
        X.append(opti.variable(nx))

    for i in range(N):
        U.append(opti.variable(nq))

    for i in range(N):
        # DYNAMIC CONSTRAINT
        opti.subject_to( X[i+1] == X[i] + dt * f(X[i], U[i]) )
        
        # JOINT POSITION AND VELOCITY BOUNDS
        opti.subject_to( opti.bounded( lbx, X[i], ubx ) )
        
        # JOINT TORQUES BOUNDS
        opti.subject_to( opti.bounded( tau_min, inv_dyn(X[i], U[i]), tau_max ) )
    
    
    # SET INITIAL STATE
    opti.subject_to( X[0] == param_x_init )
    opti.subject_to( X[N-1] == X[N] ) 

    cost = 1
    opti.minimize(cost)

    opts = {
        "ipopt.print_level": 0,
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-6,
        "ipopt.compl_inf_tol": 1e-6,
        "print_time": 0,               
        "detect_simple_bounds": True
    }
    opti.solver("ipopt", opts)

    q0 = np.random.uniform(robot.model.lowerPositionLimit, robot.model.upperPositionLimit, nq)   # initial joint configuration
    dq0 = np.random.uniform(-robot.model.velocityLimit, robot.model.velocityLimit, nq)   # initial joint velocities
    opti.set_value(param_x_init, np.concatenate([q0, dq0])) # x = [q,dq]
    try:
        sol = opti.solve()
    except:
        sol = opti.debug
    
    if (sol.stats()["return_status"] == 'Solve_Succeeded'):
        num_positive += 1
        if num_positive <= TOT_POS:
            df.loc[len(df)] = [q0[0], q0[1], q0[2], q0[3], q0[4], q0[5], dq0[0], dq0[1], dq0[2], dq0[3], dq0[4], dq0[5], 1]
            progress_bar.update(1)
    else:
        num_negative += 1
        if num_negative <= TOT_NEG:
            df.loc[len(df)] = [q0[0], q0[1], q0[2], q0[3], q0[4], q0[5], dq0[0], dq0[1], dq0[2], dq0[3], dq0[4], dq0[5], 0]
            progress_bar.update(1)

    # Update the description of the progress bar
    progress_bar.set_description(f"Positive: {num_positive}/{TOT_POS} Negative: {num_negative}/{TOT_NEG}")

    if num_positive >= TOT_POS and num_negative >= TOT_NEG:
        done = True

progress_bar.close()
name = 'dataset_' + str(N) + '.csv'
df.to_csv(name, index=False)    
        