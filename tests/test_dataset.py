import torch
from pathlib import Path
from src.data.dataset import NEODataset

PARQUET = Path.cwd() / "data" / "neos_ml.parquet"


def test_len_matches_parquet():
    ds = NEODataset(PARQUET)
    assert len(ds) == 61877

def test_getitem_returns_feature_tensor_and_scalar_label():
    ds = NEODataset(PARQUET)
    features, label = ds[0]

    # features should be a 1-D float tensor, one value per feature column
    assert isinstance(features, torch.Tensor)
    assert features.dtype == torch.float32
    assert features.ndim == 1

    # label should be a single 0 or 1
    assert int(label) in (0, 1)

def test_features_exclude_target():
    ds = NEODataset(PARQUET)
    features, _ = ds[0]
    # full parquet has N columns; features must be N-1 (target dropped)
    assert features.shape[0] == ds.NEOS.shape[1] - 1