import numpy as np
import pandas as pd
import spacy
import mapply
from tqdm import tqdm
# from fancyimpute import (KNN, 
#                         NuclearNormMinimization, 
#                         SoftImpute, 
#                         BiScaler, 
#                         # IterativeImputer
#                         )
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import (StandardScaler, 
                                  RobustScaler,
                                  OneHotEncoder,
                                  MultiLabelBinarizer
                                  )
from sklearn.impute import (SimpleImputer,
                            IterativeImputer
                            )
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from scipy.sparse import csr_matrix

TRAIN_SIZE = 33538
mapply.init()

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
    nlp = spacy.load("en_core_web_sm")
    def ner_extraction(s):
        temp = nlp(s).ents
        return ' '.join([ent.label_ for ent in temp])
    # tfidf vectorizer performs worse for some reason
    entities_listing = text_cleaned['description'].mapply(lambda s: ner_extraction(s))
    entities_listing = csr_matrix.toarray(CountVectorizer().fit_transform(entities_listing))
    entities_host = text_cleaned['host_about'].mapply(lambda s: ner_extraction(s))
    entities_host = csr_matrix.toarray(CountVectorizer().fit_transform(entities_host))
    def sentiment_analysis(s):
        ana = SentimentIntensityAnalyzer()
        vader = ana.polarity_scores(s)
        blob = TextBlob(s)
        return [
                vader['neg'], vader['neu'], vader['pos'], 
                blob.sentiment.polarity, blob.sentiment.subjectivity,
            ]
    lst = np.array(text_cleaned['description'].mapply(lambda s: sentiment_analysis(s)).values.tolist())
    train = lst[:TRAIN_SIZE,:]
    text = KMeans(clusters).fit(train).predict(lst).reshape(-1, 1)
    text = OneHotEncoder(sparse=False).fit_transform(text)

    text = np.concatenate([
        text_quant, 
        entities_listing, 
        entities_host,
        text,
    ], axis=1)
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

