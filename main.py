import numpy as np
import pandas as pd
import os
import os.path as osp
from cleaning import *
from preprocessing import *
from model import *
from postprocessing import *
from tqdm import tqdm

# Use these three serials to achieve the best result. 
# There should be four, but I can't find the exact last serial.
# This should achieve a similar result.
SERIALS = [59523, 36124, 29041]
NUMERIC_TAIL = 99.9
NUMERIC_STD = 'standard'
CAT_STRATEGY = 'constant'
CAT_FILL = 'unk'
PRE_PRICE = False
RETRAIN_TEXT = True
TEXT_CLUSTERS = 20 # 15

# tofill


def main():
    if not osp.exists('data'): os.makrdirs('data')
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
    cats = cat_transform(cat_cleaned, strategy=CAT_STRATEGY, value=CAT_FILL)
    print('done!')
    print('preprocessing numerics...', end='')
    numeric_cleaned = numeric_clean(df, upper=NUMERIC_TAIL)
    numerics = numeric_transform(numeric_cleaned, std_mode=NUMERIC_STD)
    print('done!')
    print('preprocessing mutilabels...', end='')
    multilabel_cleaned = multilabel_clean(df)
    multilabels = multilabel_transform(multilabel_cleaned)
    print('done!')
    print('preprocessing texts...', end='')
    if osp.exists('saved_text_features.csv') and not RETRAIN_TEXT:
        texts = pd.read_csv('saved_text_features.csv').values
    else:
        text_cleaned = text_clean(df)
        texts = text_transform(text_cleaned, clusters=TEXT_CLUSTERS)
    print('done!')
    if PRE_PRICE:
        print('preprocessing past prices...', end='')
        prices = price_transform(df)
        print('done!')

    print('preparing dataset for train and prediction...', end='')
    if PRE_PRICE:
        X, y = np.concatenate([
            bools, 
            cats, 
            multilabels, 
            numerics, 
            texts, 
            prices,
        ], axis=1),df['price'].values
    else:
        X, y = np.concatenate([
            bools, 
            cats, 
            multilabels, 
            numerics, 
            texts, 
        ], axis=1),df['price'].values
    X_train_all, y_train_all = X[:train_size,:], y[:train_size]
    X_test = X[train_size:,:]

    for serial in tqdm(SERIALS):
        X_train, X_val, y_train, y_val = split_data(X_train_all, y_train_all, serial=serial) #serial here 55931
        train_set, val_set = get_dataset(X_train, y_train, X_val, y_val)
        print('done!')

        params = {
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": "rmse",
                'max_depth': 24, #24
                "num_leaves": 45, #45
                "learning_rate": 0.005, #0.005
                "feature_fraction": 0.2, #0.2
                "bagging_fraction": 0.5, #0.5
                "min_split_gain": 0.5,
                "min_child_weight": 1,
                "min_child_samples": 5,
                "n_estimators": 20000, #20000
                "verbose": -1,
                "reg_lambda": 0.005,
                # "reg_alpha": 0.005,
                "n_jobs": -1,
                "early_stopping_rounds": 2000,
                }

        print('start training')
        model = train(params, train_set, val_set)
        y_pred = predict(model, X_test)
        filename = osp.join('data', f'{serial}.csv')
        filename = write_csv(test_df, y_pred, filename=filename) #serial here
        print(f'inference for test data finished, predictions stored as {filename}')
    pd.DataFrame(SERIALS).to_csv(osp.join('data', 'serials.csv'))
    print('start postprocessing...', end='')
    execute(not PRE_PRICE)

if __name__ == "__main__":
    main()
