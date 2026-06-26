import torch
import torch.nn as nn
import pyarrow.parquet as pq

class NEOModel(nn.Module):
    def __init__(self, input_len, hidden_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_len, hidden_size)
        self.ReLU = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.ReLU(x)
        x = self.layer_2(x)
        x = self.Sigmoid(x)
        return(x)