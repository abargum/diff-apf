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

class AllPass1stCell_W(Module):
    def __init__(self):
        super(AllPass1stCell_W, self).__init__()
        self.trainable_vector = Parameter(torch.zeros(1024))
        self.trainable_vector.requires_grad = False
        self.bias = Parameter(torch.rand(1024))
        
        self.dense_layer_1 = torch.nn.Linear(1024, 512)
        self.tanh1 = Sine()
        self.dense_layer_2 = torch.nn.Linear(512, 256)
        self.tanh2 = Sine()
        self.dense_layer_3 = torch.nn.Linear(256, 128)
        self.tanh3 = Sine()
        self.dense_layer_4 = torch.nn.Linear(128, 2)
        self.tanh4 = torch.nn.Tanh()
        
        self.a = torch.tensor([0])
        self.g = torch.tensor([0])

    def init_states(self, size):
        state = torch.zeros(size).to(next(self.parameters()).device)
        return state

    def forward(self, input, state):
        x = self.trainable_vector + self.bias
        x = self.dense_layer_1(x)
        x = self.tanh1(x)
        x = self.dense_layer_2(x)
        x = self.tanh2(x)
        x = self.dense_layer_3(x)
        x = self.tanh3(x)
        x = self.dense_layer_4(x)
        x = self.tanh4(x)
        
        self.a = x[0]
        self.g = x[1] 
        
        output = ( (self.a + self.g) * input + state) / (1.0 + self.a * self.g)
        state = (1.0 + self.a * self.g) * input - (self.a + self.g) * output
        return output, state

class AllPass1stOrder_W(Module):
    def __init__(self):
        super(AllPass1stOrder_W, self).__init__()
        self.cell = AllPass1stCell_W()

    def forward(self, input):
        batch_size = input.shape[0]
        sequence_length = input.shape[1]
        states = self.cell.init_states(batch_size)

        out_sequence = torch.zeros(input.shape[:-1]).to(input.device)
        for s_idx in range(sequence_length):
            out_sequence[:, s_idx], states = self.cell(input[:, s_idx].view(-1), states)
        out_sequence = out_sequence.unsqueeze(-1)

        return out_sequence