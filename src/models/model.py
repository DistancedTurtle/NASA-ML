import torch.nn as nn


class NEOModel(nn.Module):
    def __init__(self, input_len, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.model(x)
