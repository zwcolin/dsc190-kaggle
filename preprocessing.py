import numpy as np
import pandas as pd
from tqdm import tqdm
from fancyimpute import (KNN, 
                        NuclearNormMinimization, 
                        SoftImpute, 
                        BiScaler, 
                        # IterativeImputer
                        )
from sklearn.preprocessing import (StandardScaler, 
                                  RobustScaler,
                                  OneHotEncoder,
                                  MultiLabelBinarizer
                                  )
from sklearn.impute import (SimpleImputer,
                            IterativeImputer
                            )
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor, \
ARDRegression, LinearRegression, ElasticNet, ElasticNetCV, PassiveAggressiveRegressor, RidgeClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


TRAIN_SIZE = 33538

def standardization_quant(df, std_mode='standard'):
    train = df[:TRAIN_SIZE,:]
    if std_mode=='standard':
        return StandardScaler().fit(train).transform(df)
    elif std_mode=='robust':
        return RobustScaler().fit(train).transform(df)
    else:
        raise(NotImplementedError)

def impute_quant(data, impute_method='mice', k=5):
    if impute_method=='mice':
        train = data[:TRAIN_SIZE,:]
        return IterativeImputer().fit(train).transform(data)
    elif impute_method=='soft':
        X_incomplete_normalized = BiScaler().fit_transform(data)
        X_filled_softimpute = SoftImpute().fit_transform(X_incomplete_normalized)
        return X_filled_softimpute
    elif impute_method=='manual':
        pass
    elif impute_method=='knn':
        return KNN(k).fit_transform(data)
    else:
        raise(NotImplementedError)

def bool_transform(bool_cleaned):
    return bool_cleaned.values

def numeric_transform(numeric_cleaned, std_mode='standard',
                        impute_method='mice', k=5):
    numeric_cleaned = np.array(numeric_cleaned)
    numeric = impute_quant(numeric_cleaned, impute_method, k)
    numeric = standardization_quant(numeric, std_mode)
    return numeric

def cat_transform(cat_cleaned, strategy='constant', value='unk'):
    train = cat_cleaned.iloc[:TRAIN_SIZE,:]
    cat = SimpleImputer(strategy=strategy, fill_value=value, verbose=1).fit(train).transform(cat_cleaned)
    cat = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(cat[:TRAIN_SIZE,:]).transform(cat)
    return cat

def multilabel_transform(mutilabel_cleaned):
    train = mutilabel_cleaned.iloc[:TRAIN_SIZE,:]
    amen = MultiLabelBinarizer().fit(train['amenities']).transform(mutilabel_cleaned['amenities'])
    host = MultiLabelBinarizer().fit(train['host_verifications']).transform(mutilabel_cleaned['host_verifications'])
    return np.concatenate([amen, 
        host
        ], axis=1)

def text_transform(text_cleaned, clusters=10, std_mode='standard'):
    text_na = text_cleaned.replace('<unk>', np.nan).isna().sum(axis=1).values.reshape(-1, 1)
    text_len = text_cleaned.apply(lambda se: se.apply(lambda s: len(s.split(' ')))).values
    text_quant = np.concatenate([text_na, text_len], axis=1)
    text_quant = standardization_quant(text_quant, std_mode)
    lst = []
    ana = SentimentIntensityAnalyzer()
    for i in tqdm(range(text_cleaned['description'].shape[0])):
        scores = ana.polarity_scores(text_cleaned['description'][i])
        lst.append([scores['neg'], scores['neu'], scores['pos']])
    lst = np.array(lst)
    train = lst[:TRAIN_SIZE,:]
    pd.DataFrame(lst).to_csv('coordianates_text_features.csv', index=False)
    text = KMeans(clusters).fit(train).predict(lst).reshape(-1, 1)
    # text = AgglomerativeClustering(clusters).fit_predict(lst).reshape(-1, 1)
    text = OneHotEncoder(sparse=False).fit_transform(text)
    text = np.concatenate([text_quant, text], axis=1)
    pd.DataFrame(text).to_csv('saved_text_features.csv', index=False)
    return text

def price_transform(df):
    price_mask = df['price'].groupby(df['host_id']).transform('count')>=1
    temp = df['price'].groupby(df['host_id']).transform('mean')*price_mask.apply(lambda x: np.nan if x == 0 else x)
    test_estimates = temp[TRAIN_SIZE:].fillna(df['price'].mean())
    test_indices = test_estimates.index

    price_mask = df['price'].groupby(df['host_id']).transform('count')>=2
    temp = df['price'].groupby(df['host_id']).transform('mean')*price_mask.apply(lambda x: np.nan if x == 0 else x)
    train_estimates = temp[:TRAIN_SIZE].fillna(df['price'].mean())
    train_indices = train_estimates.index

    estimates = pd.concat([train_estimates, test_estimates]).values.reshape(-1, 1)
    return estimates

