import sys

import numpy as np
import xgboost as xgb

from src.models.tree import train_tree

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss"}


def make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 4)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)
    return X, y


def returns_booster():
    X, y = make_data()
    model = train_tree(X, y, X, y, PARAMS)
    assert isinstance(model, xgb.Booster)


def predicts_probabilities():
    X, y = make_data()
    model = train_tree(X, y, X, y, PARAMS)
    preds = model.predict(xgb.DMatrix(X))

    assert preds.shape == (len(y),)
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0


def learns_separable_pattern():
    X, y = make_data()
    model = train_tree(X, y, X, y, PARAMS)
    preds = (model.predict(xgb.DMatrix(X)) >= 0.5).astype(np.float32)

    assert (preds == y).mean() > 0.9


if __name__ == "__main__":
    globals()[sys.argv[1]]()
