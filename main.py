import numpy as np
import pandas as pd
from cleaning import *
from preprocessing import *
from model import *

SERIAL = 'unknown'
FILENAME = 'submission.csv'
NUMERIC_TAIL = 99.9
NUMERIC_STD = 'standard'
CAT_STRATEGY = 'constant'
CAT_FILL = 'unk'
# tofill


def main():
    print('reading and parsing raw data...', end='')
    train_df = pd.read_csv("train.csv", low_memory=False)
    test_df = pd.read_csv("test.csv", low_memory=False)
    train_size = train_df.shape[0]
    test_size = test_df.shape[0]
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    print('done!')

    print('preprocessing booleans...', end='')
    bool_cleaned = bool_clean(df)
    bools = bool_transform(bool_cleaned)
    print('done!')
    print('preprocessing one-hots...', end='')
    cat_cleaned = cat_clean(df)
    cats = cat_transform(cat_cleaned)
    print('done!')
    print('preprocessing numerics...', end='')
    numeric_cleaned = numeric_clean(df, upper=NUMERIC_TAIL)
    numerics = numeric_transform(numeric_cleaned)
    print('done!')
    print('preprocessing mutilabels...', end='')
    multilabel_cleaned = multilabel_clean(df)
    multilabels = multilabel_transform(multilabel_cleaned)
    print('done!')
    print('preprocessing texts...', end='')
    text_cleaned = text_clean(df)
    texts = text_transform(text_cleaned, clusters=15)
    print('done!')

    print('preparing dataset for train and prediction...', end='')
    X, y = np.concatenate([bools, cats, multilabels, texts, numerics], axis=1),\
            df['price'].values
    X_train_all, y_train_all = X[:train_size,:], y[:train_size]
    X_test = X[train_size:,:]
    X_train, X_val, y_train, y_val = split_data(X_train_all, y_train_all) #serial here
    train_set, val_set = get_dataset(X_train, y_train, X_val, y_val)
    print('done!')

    params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 45,
            "learning_rate": 0.005,
            "feature_fraction": 0.2,
            "bagging_fraction": 0.55,
            "min_split_gain": 0.5,
            "min_child_weight": 1,
            "min_child_samples": 5,
            "n_estimators": 30000,
            "verbose": -1,
            "reg_lambda": 0.005,
            "reg_alpha": 0.005,
            "n_jobs": -1,
            "early_stopping_rounds": 2000,
            }

    print('start training')
    model = train(params, train_set, val_set)
    y_pred = predict(model, X_test)
    filename = write_csv(test_df, y_pred, filename=FILENAME) #serial here
    print('inference for test data finished, predictions stored as {}'.format(filename))

if __name__ == "__main__":
    main()
