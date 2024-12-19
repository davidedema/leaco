import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
from termcolor import colored
from dpendulum import DPendulum

from example_robot_data.robots_loader import load, load_full



def main():
    
    nq=51   # number of discretization steps for the joint angle q
    nv=21   # number of discretization steps for the joint velocity v
    nu=11   # number of discretization steps for the joint torque u
    
    
    # parse the arguments
    N_sim = 100
    print("Load robot model")
    env = DPendulum(nq,nv,nu)

    # print("Create KinDynComputations object")
    # joints_name_list = [s for s in robot.model.names[1:]]   # skip the first name because it is "universe"
    # nq = len(joints_name_list)                              # number of joints
    # nx = 2*nq                                               # size of the state variable
    # kinDyn = KinDynComputations(urdf, joints_name_list)

    # DO_WARM_START = True
    # SOLVER_TOLERANCE = 1e-4
    # SOLVER_MAX_ITER = 3

    # SIMULATOR = "pinocchio"
    # POS_BOUNDS_SCALING_FACTOR = 0.2
    # VEL_BOUNDS_SCALING_FACTOR = 2.0 
    # qMin = POS_BOUNDS_SCALING_FACTOR * robot.model.lowerPositionLimit
    # qMax = POS_BOUNDS_SCALING_FACTOR * robot.model.upperPositionLimit
    # vMax = VEL_BOUNDS_SCALING_FACTOR * robot.model.velocityLimit
    # tauMin = -robot.model.effortLimit 
    # tauMax = robot.model.effortLimit

    # dt = 0.010          # time step MPC

    # q0 = np.zeros(nq)   # initial joint configuration
    # dq0= np.zeros(nq)   # initial joint velocities

    # p_ee_des = np.array([-0.1, 0.1, -0.6]) # desired end-effector position

    # wall_y = 0.05      # y position of the wall

    # w_p = 1e2           # position weight
    # w_a = 1e-5          # acceleration weight

    # r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
    # simu = RobotSimulator(conf_ur5, r)
    # simu.init(q0, dq0)
    # simu.display(q0)
        


    # print("Create optimization parameters")
    # ''' The parameters P contain:
    #     - the initial state (first 12 values)
    #     - the target configuration (last 6 values)
    # '''
    # opti = cs.Opti()
    # param_x_init = opti.parameter(nx)
    # param_p_ee_des = opti.parameter(3)
    # cost = 1

    # # create the dynamics function
    # q   = cs.SX.sym('q', nq)
    # dq  = cs.SX.sym('dq', nq)
    # ddq = cs.SX.sym('ddq', nq)
    # state = cs.vertcat(q, dq)
    # rhs    = cs.vertcat(dq, ddq)
    # f = cs.Function('f', [state, ddq], [rhs])

    # # create a Casadi inverse dynamics function
    # H_b = cs.SX.eye(4)     # base configuration
    # v_b = cs.SX.zeros(6)   # base velocity
    # bias_forces = kinDyn.bias_force_fun()
    # mass_matrix = kinDyn.mass_matrix_fun()
    # # discard the first 6 elements because they are associated to the robot base
    # h = bias_forces(H_b, q, v_b, dq)[6:]
    # M = mass_matrix(H_b, q)[6:,6:]
    # tau = M @ ddq + h
    # inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

    # # pre-compute state and torque bounds
    # lbx = qMin.tolist() + (-vMax).tolist()
    # ubx = qMax.tolist() + vMax.tolist()
    # tau_min = (tauMin).tolist()
    # tau_max = (tauMax).tolist()

    # # create all the decision variables
    # X, U = [], []
    # X += [opti.variable(nx)] # do not apply pos/vel bounds on initial state
    # for k in range(1, N+1): 
    #     X += [opti.variable(nx)]
    #     opti.subject_to( opti.bounded(lbx, X[-1], ubx) )
    # for k in range(N): 
    #     U += [opti.variable(nq)]

    # print("Add initial conditions")
    # opti.subject_to(X[0] == param_x_init)

    # opti.minimize(cost)

    # print("Create the optimization problem")
    # opts = {
    #     "error_on_fail": False,
    #     "ipopt.print_level": 0,
    #     "ipopt.tol": SOLVER_TOLERANCE,
    #     "ipopt.constr_viol_tol": SOLVER_TOLERANCE,
    #     "ipopt.compl_inf_tol": SOLVER_TOLERANCE,
    #     "print_time": 0,             
    #     "detect_simple_bounds": True,
    #     "ipopt.max_iter": 1000
    # }
    # opti.solver("ipopt", opts)

    # # Solve the problem to convergence the first time
    # x = np.concatenate([q0, dq0])
    # opti.set_value(param_p_ee_des, p_ee_des)
    # opti.set_value(param_x_init, x)
    # sol = opti.solve()
    # opts["ipopt.max_iter"] = SOLVER_MAX_ITER
    # opti.solver("ipopt", opts)
