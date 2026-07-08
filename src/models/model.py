import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features), 
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        return self.block(x)


class NEOModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.block1 = MLPBlock(input_size, hidden_size)
        self.block2 = MLPBlock(hidden_size, 130)
        self.fc1 = nn.Linear(130, 1)


    def forward(self, x):
        return self.fc1(self.block2(self.block1(x)))
    
