import torch
from torch.utils.data import Dataset
import pandas as pd

TARGET_COLUMN = "is_potentially_hazardous_asteroid"


class NEODataset(Dataset):
    def __init__(self, parquet_file, transform=None, target_transform=None):
        df = pd.read_parquet(parquet_file)
        features = df.drop(columns=TARGET_COLUMN).values
        labels = df[TARGET_COLUMN].values
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform
        self.target_transform = target_transform
        self.error_log = []

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        try:
            features = self.features[idx]
            label = self.labels[idx]

            if self.transform:
                features = self.transform(features)
            if self.target_transform:
                label = self.target_transform(label)

            return features, label

        except Exception as e:
            self.error_log.append({
                "index": idx, 
                "error": str(e),
            })
            print(f"skipping image {idx}: {e}")
            next_idx = (idx+1) % len(self)
            return self.__getitem__(next_idx)

    def get_error_summary(self):
        if not self.error_log:
            print("no errors encountered")
        else: 
            for error in self.error_log[:5]:
                print(f"Index {error['index']}: {error['error']}")
            if len(self.error_log) > 5:
                print(f"...{len(self.error_log) - 5 } additional errors")



def get_normalization_transform(mean, std):
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
    std_tensor = torch.as_tensor(std, dtype=torch.float32).clone()

    std_tensor[std_tensor == 0.0] = 1.0 

    def normalize(tensor):
        standardized = (tensor - mean_tensor) / std_tensor
        return torch.nan_to_num(standardized, nan=0.0)

    return normalize
