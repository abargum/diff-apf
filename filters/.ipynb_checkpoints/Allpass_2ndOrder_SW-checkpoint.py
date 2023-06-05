import torch
from torch.nn import Module, Parameter
from torch import FloatTensor
import numpy as np

device = torch.device('cuda')
torch.set_default_tensor_type(torch.DoubleTensor)

class Sine(Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class Denormalize(Module):
    def __init__(self, val_min, val_max):
        super().__init__()
        self.val_min = val_min
        self.val_max = val_max
    def forward(self, x):
        return ((self.val_max - self.val_min) / 2.0) * x + ((self.val_max + self.val_min) / 2.0)

class AllPass2ndCell_SW(Module):
    def __init__(self):
        super(AllPass2ndCell_SW, self).__init__()
        
        self.trainable_vector = Parameter(torch.zeros(1024))
        self.trainable_vector.requires_grad = False
        self.bias = Parameter(torch.rand(1024))
        
        self.dense_layer_1 = torch.nn.Linear(1024, 512)
        self.tanh1 = Sine()
        self.dense_layer_2 = torch.nn.Linear(512, 256)
        self.tanh2 = Sine()
        self.dense_layer_3 = torch.nn.Linear(256, 128)
        self.tanh3 = Sine()
        self.dense_layer_4 = torch.nn.Linear(128, 3)
        self.tanh4 = torch.nn.Tanh()
        
        self.c = torch.tensor([0])
        self.d = torch.tensor([0])
        self.a = torch.tensor([0])
        
        self.R = torch.tensor([0])
        self.fc = torch.tensor([0])
        
        self.denormalize = Denormalize(val_min = 20, val_max = 20000)
    
    def _cat(self, vectors):
        return torch.cat([v_.unsqueeze(-1) for v_ in vectors], dim=-1)

    def init_states(self, size):
        v = torch.zeros(size, 10).to(next(self.parameters()).device)
        return v

    def forward(self, input, v):    
        x = self.trainable_vector + self.bias
        x = self.dense_layer_1(x)
        x = self.tanh1(x)
        x = self.dense_layer_2(x)
        x = self.tanh2(x)
        x = self.dense_layer_3(x)
        x = self.tanh3(x)
        x = self.dense_layer_4(x)
        x = self.tanh4(x)
        
        self.R = x[0]
        self.fc = self.denormalize(x[1])
        
        self.c = self.R ** 2
        self.d = -2.0 * self.R * torch.cos(2 * np.pi * self.fc/192000)
        self.a = x[2]
        
        output = (input * (self.c + self.a * self.a + self.a * self.d) + v[:, 3]) / (1.0 + self.a * self.a * self.c + self.a * self.d)
        
        v[:, 3] = v[:, 2]
        v[:, 2] = v[:, 1]
        v[:, 1] = v[:, 0]
        v[:, 0] = v[:, 8]
        
        v[:, 8] = input * (self.a + self.a + self.d + self.a * self.c) + \
        output * (-self.a * self.c - self.a * self.c - self.d - self.a) + \
        (self.a * (input - self.c * output)) * (-self.a * self.a) + (-self.a * self.a + 1.0) * v[:, 7].clone()
        
        old_state = torch.zeros((v[:,9].size())).to(device)
        old_state[:] = v[:, 9]
        
        v[:, 9] = input - self.c * output - self.a * (self.a * (input - self.c * output) + v[:, 7])
        
        v = self._cat([ v[:, 0], v[:, 1], v[:, 2], v[:, 3], old_state, v[:, 4], v[:, 5], v[:, 6], v[:, 8], v[:, 9]])
        
        return output, v

class AllPass2ndOrder_SW(Module):
    def __init__(self):
        super(AllPass2ndOrder_SW, self).__init__()
        self.cell = AllPass2ndCell_SW()

    def forward(self, input):
        batch_size = input.shape[0]
        sequence_length = input.shape[1]
        states = self.cell.init_states(batch_size)

        out_sequence = torch.zeros(input.shape[:-1]).to(input.device)
        for s_idx in range(sequence_length):
            out_sequence[:, s_idx], states = self.cell(input[:, s_idx].view(-1), states)
        out_sequence = out_sequence.unsqueeze(-1)

        return out_sequence