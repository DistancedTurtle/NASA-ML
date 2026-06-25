import torch
import torch.optimizers as optim
import torch.nn as nn

class NEOModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear()
