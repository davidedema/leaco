import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
from termcolor import colored

from example_robot_data.robots_loader import load, load_full



def main():
    
    # parse the arguments
    N_sim = 100
    print("Load robot model")
    robot = load("double_pendulum")
    
    print("Create KinDynComputations object")
    joints_name_list = [s for s in robot.model.names[1:]]
    
    nq = len(joints_name_list)  # number of joints
    nx = nq * 2

    kinDyn = KinDynComputations(robot.urdf, joints_name_list)
    
    dt = 0.01           # optimal control
    N = 60              # time horizon 
    
    # TODO: HERE WE HAVE TO PUT SAMPLES FOR THE INITIAL CONFIG
    # q0 = np.ones(nq)   # initial joint configuration
    # dq0= np.ones(nq)   # initial joint velocities
    
    # qdes = np.array([0,0]) # desired joint config
    
    # create the optimal control problem 
    
    opti = cs.Opti()
    param_x_init = opti.parameter(nx)   # variable with a fixed value
    # param_q_des = opti.parameter(nq)    # variable with a fixed value
    
    #* CLASSIC Multi Body Dynamics Modeling
    # M(q) * ddq + h(q, dq) = tau   <- inverse dynamics (RNEA)
    # u = tau
    # x = (q, dq)
    # dx = f(x, u)          
    # ddq = M^-1 (u - h(x))         <- forward dynamics (ABA)
    
    #* ALTERNATIVE Multi Body Dynamics Modeling
    # x = (q, dq)
    # u = ddq
    # dx = f(x, u) -> double integrator (Linear Dynamical System)
    # Torques are no longer a variable of OCP so Tbounds become:
    #           tau_min <= M(q) * u + h(q, dq) <= tau_max
    
    # create the dynamics function
    q   = cs.SX.sym('q', nq)
    dq  = cs.SX.sym('dq', nq)
    ddq = cs.SX.sym('ddq', nq)
    state = cs.vertcat(q, dq)
    rhs    = cs.vertcat(dq, ddq)        # right hand side of dynamics
    # funciton f takes as input (x, u = ddq) and compute dx=(dq, ddq)
    f = cs.Function('f', [state, ddq], [rhs])
    
    # create the inverse dynamics function
    H_b = np.eye(4)         # transf mat for base (but is fixed so eye)
    v_b = np.zeros(6)       # velocity of the base is 0 since fixed base
    bias_forces = kinDyn.bias_force_fun()
    mass_mat = kinDyn.mass_matrix_fun()
    # remove first 2 since associated to robot base
    h = bias_forces(H_b, q, v_b, dq)[6:]
    M = mass_mat(H_b, q)[6:, 6:]
    
    tau = M @ ddq + h
    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])
    
    # pre-compute state and torque bounds
    lbx = robot.model.lowerPositionLimit.tolist() + (-robot.model.velocityLimit).tolist()
    ubx = robot.model.upperPositionLimit.tolist() + robot.model.velocityLimit.tolist()
    tau_min = (-robot.model.effortLimit).tolist()
    tau_max = robot.model.effortLimit.tolist()
    
    # create decsion variables
    X, U = [], []
    for i in range(N+1):
        X.append(opti.variable(nx))
    
    for i in range(N):
        U.append(opti.variable(nq))
    
    cost = 1
    
    for i in range(N):
        
        # Explicit euler for integrating dyn
        # x_next = x + dt * f(x,u)
        opti.subject_to( X[i+1] == X[i] + dt * f(X[i], U[i]) )  # dyn constraints
    
    opti.subject_to(X[N][nq:] == 0.0)       # constraint on the final state
    opti.minimize(cost)
    
    
    print("Create the optimization problem")
    
    opts = {
        "ipopt.print_level": 0,
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-6,
        "ipopt.compl_inf_tol": 1e-6,
        "print_time": 0,                # print information about execution time
        "detect_simple_bounds": True
    }
    
    opti.solver("ipopt", opts)
    
    print("start solving the problem")
    # opti.set_value(param_q_des, qdes)
    N_TRY = 50
    for i in range(N_TRY):
        q0 = np.random.uniform(robot.model.lowerPositionLimit, robot.model.upperPositionLimit, nq)   # initial joint configuration
        dq0 = np.random.uniform(-robot.model.velocityLimit, robot.model.velocityLimit, nq)   # initial joint velocities
        opti.set_value(param_x_init, np.concatenate([q0, dq0])) # constraint on the initial state
        sol = opti.solve()
        print(sol.stats()["return_status"])