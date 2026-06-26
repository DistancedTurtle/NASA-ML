import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from pathlib import Path

parquet_file = Path.cwd() / "data" / "neos_ml.parquet"

class NEODataset(Dataset):
    def __init__(self, parquet_file, transform=None, target_transform=None):
        self.NEOS = pd.read_parquet(parquet_file)
        self.labels = self.NEOS.iloc[:, 1]
        self.features = self.NEOS.drop(columns='is_potentially_hazardous_asteroid')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.NEOS)

    def __getitem__(self, idx):
        features = self.features.iloc[idx, :]
        label = self.labels.iloc[idx]
        features = torch.tensor(features.values, dtype=torch.float32)

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
            
        return features, label

def get_normalization_transform(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])



