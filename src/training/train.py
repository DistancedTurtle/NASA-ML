import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import os
import pandas as pd
from pathlib import Path
from data.dataset import NEODataset, get_normalization_transform
from models.model import NEOModel

parquet_file = Path.cwd() / "data" / "neos_ml.parquet"
data_dir = Path.cwd() / "data"

df = pd.read_parquet(parquet_file)

tensor_data = torch.tensor([df.values], torch.float32)

means = torch.mean(X, dim=0)
stds = torch.std(X, dim=0)

transforms = get_normalization_transform(means, stds)

data = NEODataset(parquet_file, data_dir, transform = transforms)

model = NEOModel()
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model.to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 5


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = loss_fn(y_hat, )
    loss.backward()
    optimizer.step()
    
    
