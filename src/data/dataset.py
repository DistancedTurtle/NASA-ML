import torch
from torch.utils.data import Dataset
import pandas as pd

TARGET_COLUMN = "is_potentially_hazardous_asteroid"


class NEODataset(Dataset):
    def __init__(self, parquet_file, transform=None, target_transform=None):
        df = pd.read_parquet(parquet_file)

        # Convert once, up front, to plain tensors. __getitem__ then just
        # indexes these -- no per-access pandas/numpy/tensor churn.
        features = df.drop(columns=TARGET_COLUMN).values
        labels = df[TARGET_COLUMN].values
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        return features, label


def get_normalization_transform(mean, std):
    # as_tensor avoids a copy/warning when mean/std are already tensors;
    # std is cloned because we mutate it below.
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
    std_tensor = torch.as_tensor(std, dtype=torch.float32).clone()

    std_tensor[std_tensor == 0.0] = 1.0  # avoid divide-by-zero on constant cols

    def normalize(tensor):
        standardized = (tensor - mean_tensor) / std_tensor
        # Missing values (NaN) become 0 after standardization, i.e. imputed
        # with the (training) mean. nan_to_num also clears any inf.
        return torch.nan_to_num(standardized, nan=0.0)

    return normalize
