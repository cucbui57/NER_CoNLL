import torch
import torch.nn as nn
import numpy as np


class highway(nn.Module):
    def __init__(self, input_size, num_layers=1, dropout=0.1):
        super(highway, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.transform_gate = nn.ModuleList()
        self.carry_gate = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        for i in range(num_layers):
            tmp_transform_gate = nn.Linear(input_size, input_size)
            tmp_carry_gate = nn.Linear(input_size, input_size)
            self.transform_gate.append(tmp_transform_gate)
            self.carry_gate.append(tmp_carry_gate)

    def forward(self, x):
        # t = nn.functional.sigmoid(self.transform_gate[0](x))
        # t = torch.sigmoid(self.transform_gate[0](x))
        # h = torch.relu(self.carry_gate[0](x))
        # x = t * h + (1 - t) * x

        for i in range(0, self.num_layers):
            t = torch.sigmoid(self.transform_gate[i](x))
            h = torch.relu(self.carry_gate[i](x))
            h = self.dropout(h)
            x = t * h + (1 - t) * x
        return x
