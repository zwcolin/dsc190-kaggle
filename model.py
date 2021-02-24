import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def split_data(X, y, proportion=0.1, serial=55931):
    X_train, X_val, y_train, y_val = \
    train_test_split(X, y, test_size=0.1, random_state=serial)
    return (X_train, X_val, y_train, y_val)

def get_dataset(X_train, y_train, X_val, y_val):
    train = lgbm.Dataset(X_train, y_train)
    val = lgbm.Dataset(X_val, y_val, reference=train)
    return (train, val)

def train(params, train, val):
    model = lgbm.train(params, train, verbose_eval=50000, early_stopping_rounds= 2000,valid_sets=val)
    return model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def write_csv(test_df, y_pred, model='lgbm', serial='unknown', filename=None):
    output = pd.DataFrame()
    output['Id'] = test_df['id']
    output['Predicted'] = y_pred
    if filename == None:
        filename = '{}_{}.csv'.format(model, serial)
    else: filename = filename
    output.to_csv(filename, index=False)
    return filename
