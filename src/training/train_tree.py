import numpy as np
import pandas as pd
from pathlib import Path

from src.models.tree import train_tree

TARGET_COLUMN = "is_potentially_hazardous_asteroid"

data_dir = Path.cwd() / "data"
parquet_file = data_dir / "neos_ml.parquet"

df = pd.read_parquet(parquet_file)
features = df.drop(columns=TARGET_COLUMN).values.astype(np.float32)
labels = df[TARGET_COLUMN].values.astype(np.float32)

n = len(labels)
perm = np.random.default_rng(42).permutation(n)
train_size = int(0.7 * n)
cv_size = int((n - train_size) / 2)
train_idx = perm[:train_size]
cv_idx = perm[train_size:train_size + cv_size]

X_train, y_train = features[train_idx], labels[train_idx]
X_cv, y_cv = features[cv_idx], labels[cv_idx]

tree_params = {
    "max_depth": 6,
    "eta": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

train_tree(X_train, y_train, X_cv, y_cv, tree_params)
