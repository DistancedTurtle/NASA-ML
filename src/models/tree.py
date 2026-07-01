import xgboost as xgb


def train_tree(X_train, y_train, X_cv, y_cv, params):
    dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
    dcv = xgb.QuantileDMatrix(X_cv, label=y_cv, ref=dtrain)
    return xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dcv, "cv")],
        early_stopping_rounds=10,
        verbose_eval=10,
    )
