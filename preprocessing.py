import numpy as np
import pandas as pd
from tqdm import tqdm
from fancyimpute import (KNN, 
                        NuclearNormMinimization, 
                        SoftImpute, 
                        BiScaler, 
                        IterativeImputer)
from sklearn.preprocessing import (StandardScaler, 
                                  RobustScaler,
                                  OneHotEncoder,
                                  MultiLabelBinarizer)
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def standardization_quant(df, std_mode='standard'):
    if std_mode=='standard':
        return StandardScaler().fit_transform(df)
    elif std_mode=='robust':
        return RobustScaler().fit_transform(df)
    else:
        raise(NotImplementedError)

def impute_quant(data, impute_method='mice', k=5):
    if impute_method=='mice':
        return IterativeImputer().fit_transform(data)
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
    numeric = impute_quant(numeric_cleaned, impute_method, k)
    numeric = standardization_quant(numeric, std_mode)
    return numeric

def cat_transform(cat_cleaned, strategy='constant', value='unk'):
    cat = SimpleImputer(strategy='constant', fill_value='unk').fit_transform(cat_cleaned)
    cat = OneHotEncoder(sparse=False).fit_transform(cat)
    return cat

def multilabel_transform(mutilabel_cleaned):
    amen = MultiLabelBinarizer().fit_transform(mutilabel_cleaned['amenities'])
    host = MultiLabelBinarizer().fit_transform(mutilabel_cleaned['host_verifications'])
    return np.concatenate([amen, host], axis=1)

def text_transform(text_cleaned, clusters=10, std_mode='standard'):
    text_na = text_cleaned.replace('<unk>', np.nan).isna().sum(axis=1).values.reshape(-1, 1)
    text_len = text_cleaned.apply(lambda se: se.apply(lambda s: len(s.split(' ')))).values
    text_quant = np.concatenate([text_na, text_len], axis=1)
    text_quant = standardization_quant(text_quant, std_mode)
    lst = []
    ana = SentimentIntensityAnalyzer()
    for i in tqdm(range(text_cleaned['description'].shape[0])):
        scores = ana.polarity_scores(text_cleaned['description'][i])
        lst.append([scores['neg'], scores['neu'],scores['pos'],scores['pos']])
    text = KMeans(clusters).fit_predict(lst).reshape(-1, 1)
    text = OneHotEncoder(sparse=False).fit_transform(text)
    return np.concatenate([text_quant, text], axis=1)

