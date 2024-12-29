import torch.nn as nn
import torch

class NeuralNet(nn.Module):

    def __init__(self, in_size, hid_size, out_size):
        super(NeuralNet, self).__init__()
        
        self.linear_stack = nn.Sequential(
          nn.Linear(in_size, hid_size),
          nn.GELU(),
          nn.Linear(hid_size, hid_size),
          nn.GELU(),
          nn.Linear(hid_size, out_size),
          nn.Sigmoid()
        )
        
        self.initialize_weights()
        
    def forward(self, x):
        out = self.linear_stack(x)
        return out
    
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def create_casadi_function(self, robot_name, NN_DIR, input_size, load_weights):
        from casadi import MX, Function
        import l4casadi as l4c

        # if load_weights is True, we load the neural-network weights from a ".pt" file
        if(load_weights):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            nn_name = f'{NN_DIR}model.pt'
            nn_data = torch.load(nn_name, map_location=device)
            self.load_state_dict(nn_data['model'])

        state = MX.sym("x", input_size)        
        self.l4c_model = l4c.L4CasADi(self,
                                      device='cuda' if torch.cuda.is_available() else 'cpu',
                                      name=f'{robot_name}_model',
                                      build_dir=f'{NN_DIR}nn_{robot_name}')
        self.nn_model = self.l4c_model(state)
        # This is the function that you can use in a casadi problem
        self.nn_func = Function('nn_func', [state], [self.nn_model])