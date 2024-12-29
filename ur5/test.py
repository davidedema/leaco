from casadi import MX, Function
import torch
from model import NeuralNet
import casadi as cs
from casadi import DM

# Initialize NeuralNet
nn = NeuralNet()
nn.create_casadi_function("ur5", "/home/student/shared/orc_project/ur5/models/", 12, True)

# Test with CasADi
state = MX.sym("x", 12)  # Assuming input size of 12
output = nn.nn_func(state)       # Should return a CasADi symbolic variable

print("CasADi output:", output)


opti = cs.Opti()

opti.solver('ipopt', {'hessian_approximation': 'limited-memory',
                      'print_level': 5})

x = opti.variable(12)

opti.set_initial(x, DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) 

opti.minimize(-(x[2]+x[3]) + 10)
opti.subject_to(nn.nn_func(x) >= 0.5)

opti.solver('ipopt')

try:
    sol = opti.solve()
    print(f"Net output: {nn.nn_func(sol.value(x))}")
    
except RuntimeError as e:
    print(f"Fail: {e}")