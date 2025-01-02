import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
from termcolor import colored

import plot_utils as plut
from example_robot_data.robots_loader import load, load_full
import orc_project.double_pendulum.conf_doublep as conf_doublep
from robot_simulator import RobotSimulator
from robot_wrapper import RobotWrapper 

import torch
from model import NeuralNet

def main():
    
    # parse the arguments
    N_sim = 100
    print("Load robot model")
    robot, _, urdf, _ = load_full("double_pendulum")

    print("Create KinDynComputations object")
    joints_name_list = [s for s in robot.model.names[1:]]   # skip the first name because it is "universe"
    print(joints_name_list)
    nq = len(joints_name_list)                              # number of joints
    nx = 2*nq                                               # size of the state variable
    kinDyn = KinDynComputations(urdf, joints_name_list)
    
    # create the neural network
    net = NeuralNet()
    net.create_casadi_function("ur5", "/home/student/shared/orc_project/double_pendulum/models/", 4, True)

    DO_WARM_START = True
    SOLVER_TOLERANCE = 1e-4
    SOLVER_MAX_ITER = 3
    
    N = int(N_sim/10)
    USE_TERMINAL_CONSTRAINT = False
    THRESHOLD_CLASSIFICATION = 0.9
    
    qMin   = np.array([-np.pi,-np.pi])
    qMax   = -qMin
    vMax   = np.array([8.0,8.0])
    vMin   = -vMax
    tauMax = np.array([1.0, 1.0])
    tauMin = -tauMax

    dt_sim = 0.002

    dt = 0.010          # time step MPC

    q0 = np.zeros(nq)   # initial joint configuration
    dq0= np.zeros(nq)   # initial joint velocities
    
    qdes = qMax         # desired joint configuration, near the joint limits
    
    w_q = 1e2           # position weight
    w_a = 1e-6          # acceleration weight
    w_v = 1e-6          # velocity weight

    r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
    simu = RobotSimulator(conf_doublep, r)
    simu.init(q0, dq0)
    simu.display(q0)
        


    print("Create optimization parameters")
    ''' The parameters P contain:
        - the initial state (first 12 values)
        - the target configuration (last 6 values)
    '''
    opti = cs.Opti()
    param_x_init = opti.parameter(nx)
    param_p_ee_des = opti.parameter(3)
    cost = 0

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
    M = mass_matrix(H_b, q)[6:,6:]
    tau = M @ ddq + h
    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

    # pre-compute state and torque bounds
    lbx = qMin.tolist() + vMin.tolist()
    ubx = qMax.tolist() + vMax.tolist()
    tau_min = (tauMin).tolist()
    tau_max = (tauMax).tolist()

    # create all the decision variables
    X, U = [], []
    X += [opti.variable(nx)] # do not apply pos/vel bounds on initial state
    for k in range(1, N+1): 
        X += [opti.variable(nx)]
        opti.subject_to( opti.bounded(lbx, X[-1], ubx) )
    for k in range(N): 
        U += [opti.variable(nq)]

    print("Add initial conditions")
    opti.subject_to(X[0] == param_x_init)
    for k in range(N):     
        cost += w_q * (X[k][:nq] - qdes).T @ (X[k][:nq] - qdes)
        cost += w_v * X[k][nq:].T @ X[k][nq:]
        cost += w_a * U[k].T @ U[k]

        # print("Add dynamics constraints")
        opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))

        # print("Add torque constraints")
        opti.subject_to( opti.bounded(tau_min, inv_dyn(X[k], U[k]), tau_max))
    
    if(USE_TERMINAL_CONSTRAINT):
        opti.subject_to(net.nn_func(X[-1]) >= THRESHOLD_CLASSIFICATION)

    opti.minimize(cost)

    print("Create the optimization problem")
    opts = {
        "error_on_fail": False,
        "ipopt.print_level": 0,
        "ipopt.tol": SOLVER_TOLERANCE,
        "ipopt.constr_viol_tol": SOLVER_TOLERANCE,
        "ipopt.compl_inf_tol": SOLVER_TOLERANCE,
        "print_time": 0,             
        "detect_simple_bounds": True,
        "ipopt.max_iter": 2000,
        "ipopt.hessian_approximation": "limited-memory",        # without this parameter the l4casadi function not works
        "ipopt.nlp_scaling_method": "gradient-based",
    }
    
    opti.solver("ipopt", opts)

    # Solve the problem to convergence the first time
    x = np.concatenate([q0, dq0])
    opti.set_value(param_x_init, x)
    sol = opti.solve()
    opts["ipopt.max_iter"] = SOLVER_MAX_ITER
    opti.solver("ipopt", opts)


    comput_time = []
    tracking_err = []
    dq_l = []
    tau_l = []


    print("Start the MPC loop")
    for i in range(N_sim):
        start_time = clock()

        if(DO_WARM_START):
            # use current solution as initial guess for next problem
            for t in range(N):
                opti.set_initial(X[t], sol.value(X[t+1]))
            for t in range(N-1):
                opti.set_initial(U[t], sol.value(U[t+1]))
            opti.set_initial(X[N], sol.value(X[N]))
            opti.set_initial(U[N-1], sol.value(U[N-1]))

            # initialize dual variables
            lam_g0 = sol.value(opti.lam_g)
            opti.set_initial(opti.lam_g, lam_g0)
        
        print("Time step", i)
        opti.set_value(param_x_init, x)
        try:
            sol = opti.solve()
        except:
            # print("Convergence failed!")
            sol = opti.debug
        end_time = clock()

        print("Comput. time: %.3f s"%(end_time-start_time), 
            "Iters: %3d"%sol.stats()['iter_count'], 
            "Return status ", sol.stats()["return_status"],)


        comput_time.append(end_time-start_time)
        dq_l.append(np.linalg.norm(x[nq:]))
        
        tau = inv_dyn(sol.value(X[0]), sol.value(U[0])).toarray().squeeze()
        tau_l.append(tau[0])
        # do a proper simulation with Pinocchio
        simu.simulate(tau, dt, int(dt/dt_sim))
        x = np.concatenate([simu.q, simu.v])

        if( np.any(x[:nq] > qMax)):
            print(colored("\nUPPER POSITION LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]>qMax)[0])
        if( np.any(x[:nq] < qMin)):
            print(colored("\nLOWER POSITION LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]<qMin)[0])
            
    print("Mean computation time: %.3f s"%np.mean(comput_time))
    print("Max computation time: %.3f s"%np.max(comput_time))
    print("Mean dq_l: %.3f"%np.mean(dq_l))
    print("Max dq_l: %.3f"%np.max(dq_l))

    
if __name__ == "__main__":
    main()