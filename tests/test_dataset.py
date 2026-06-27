import pandas as pd
import pytest
import torch

from src.data.dataset import NEODataset


N_FEATURES = 3


@pytest.fixture
def sample_parquet(tmp_path):
    df = pd.DataFrame(
        {
            "absolute_magnitude_h": [10.4, 15.6, 22.1, 9.0],   # feature (col 0)
            "is_potentially_hazardous_asteroid": [0, 1, 0, 1],  # target  (col 1)
            "eccentricity": [0.22, 0.55, 0.41, 0.07],          # feature (col 2)
            "inclination": [10.8, 11.5, 9.4, 26.7],            # feature (col 3)
        }
    )
    path = tmp_path / "sample.parquet"
    df.to_parquet(path)
    return path


def test_len_matches_row_count(sample_parquet):
    ds = NEODataset(sample_parquet)
    assert len(ds) == 4


def test_getitem_returns_feature_tensor_and_scalar_label(sample_parquet):
    ds = NEODataset(sample_parquet)
    features, label = ds[0]

    # features should be a 1-D float tensor, one value per feature column
    assert isinstance(features, torch.Tensor)
    assert features.dtype == torch.float32
    assert features.ndim == 1
    assert features.shape[0] == N_FEATURES

    # label should be a single 0 or 1
    assert int(label) in (0, 1)


def test_features_exclude_target(sample_parquet):
    ds = NEODataset(sample_parquet)
    features, _ = ds[0]
    # fixture has N_FEATURES + 1 columns; the target must be dropped
    assert features.shape[0] == N_FEATURES
    # and the dataset must expose exactly N_FEATURES feature columns
    assert ds.features.shape[1] == N_FEATURES


def test_label_matches_source_row(sample_parquet):
    ds = NEODataset(sample_parquet)
    # row 1 of the fixture has target == 1; confirm the dataset returns it
    _, label = ds[1]
    assert int(label) == 1
