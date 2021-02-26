import numpy as np 
import pandas as pd
import ast 

def condense_category(col, min_freq=0.01, new_name='other'):
    series = pd.value_counts(col)
    mask = (series/series.sum()).lt(min_freq)
    return pd.Series(np.where(col.isin(series[mask].index), new_name, col))

def bool_clean(df):
    df = df.copy()
    df['host_is_superhost']\
        .fillna('f', inplace=True)
    df['host_has_profile_pic']\
        .fillna('f', inplace=True)
    df['host_identity_verified']\
        .fillna('f', inplace=True)
    df['has_review'] = df['last_review'].isna().astype(int)
    sub_df = df[[
    'host_is_superhost', 
    'host_has_profile_pic', 
    'instant_bookable', 
    'is_business_travel_ready', 
    'host_identity_verified',
    'require_guest_profile_picture', 
    'require_guest_phone_verification', 
    'has_review'
    ]]\
    .replace({'t': 1, 'f': 0})
    return sub_df

def cat_clean(df):
    df = df.copy()
    df['host_since'] = df['host_since'].apply(lambda dt: str(dt)[0:4])
    df['host_location'] = condense_category(df['host_location'], 0.004)
    df['host_response_time'] = df['host_response_time']
    df['host_neighbourhood'] = condense_category(df['host_neighbourhood'], 0.008) #0.008
    df['city'] = condense_category(df['city'])
    df['property_type'] = condense_category(df['property_type'], 
        min_freq=0.002, new_name='Other')
    df['zipcode'] = df['zipcode'].apply(lambda s: s[:5] if isinstance(s, str) else s)\
    .replace('-- de', np.nan).replace('1m', np.nan)
    df['zipcode'] = condense_category(df['zipcode'], 0.0003)
    df['room_type'] = df['room_type']
    df['bed_type'] = df['bed_type']
    df['cancellation_policy'] = condense_category(df['cancellation_policy'], 
        min_freq=0.01, new_name='other')
    sub_df = df[[
    'host_since', 
    'host_location', 
    'host_response_time', 
    'host_neighbourhood', 
    'city', 
    'zipcode', 
    'neighbourhood_cleansed', 
    'neighbourhood_group_cleansed',
    'property_type', 
    'room_type',
    'bed_type', 
    'cancellation_policy', 
    ]]
    return sub_df

def numeric_clean(df, upper=99.9):
    df = df.copy()
    df['total_na'] = df.isna().sum(axis=1)
    df['host_response_rate'] = df['host_response_rate'].apply(\
        lambda r: int(str(r)[:-1]) if isinstance(r, str) else r)
    df['host_verifications'] = df['host_verifications']\
                .apply(lambda lst: ast.literal_eval(lst))\
                .apply(lambda lst: len(lst) if isinstance(lst, list) else np.nan)
    df['beds_bedroom'] = (df['beds']/(df['bedrooms'].fillna(1).replace(0, 1)))
    df['bathrooms_bedrooms'] = (df['bathrooms']/(df['bedrooms'].fillna(1).replace(0, 1)))
    df['people_per_bed'] =  (df['accommodates'].values/(df['beds'].fillna(1).replace(0, 1)))
    df['amenities'] = df['amenities'].apply(lambda s: s\
        .replace('{', '[')\
        .replace('}', ']')\
        .replace('"', '')\
        .count(',')+1)
    df['extra_people'] = df['extra_people'].apply(lambda s: float(s[1:]))
    sub_df = df[[
    'host_response_rate', 
    'host_listings_count', 
    'host_verifications', 
    'amenities',
    'accommodates', 
    'bathrooms', 
    'bedrooms',
    'beds',
    'beds_bedroom', 
    # 'bathrooms_bedrooms', 
    # 'people_per_bed',
    'guests_included', 
    'extra_people',
    'minimum_nights', 
    'maximum_nights', 
    'number_of_reviews', 
    'reviews_per_month',
    'review_scores_rating', 
    'review_scores_accuracy',
    'review_scores_cleanliness', 
    'review_scores_checkin',
    'review_scores_communication', 
    'review_scores_location',
    'review_scores_value', 
    'total_na'
    ]].copy()
    cols = sub_df.columns
    for col in cols:
        _max = np.percentile(sub_df[col].values, upper)
        _min = sub_df[col].min()
        sub_df[col] = np.clip(sub_df[col].values, _min, _max)
    return sub_df

def multilabel_clean(df):
    df = df.copy()
    df['amenities'] = df['amenities'].apply(lambda s: s\
        .replace('{', "['")\
        .replace('}', "']")\
        .replace('"', '').replace(' ', '_').replace(',', "', '")\
        .lower())
    df['amenities'] = df['amenities'].apply(lambda lst: ast.literal_eval(lst))
    df['host_verifications'] = df['host_verifications']\
        .apply(lambda lst: ast.literal_eval(lst))\
        .apply(lambda lst: lst if isinstance(lst, list) else [])
    sub_df = df[[
    'amenities', 
    'host_verifications'
    ]]
    return sub_df

def text_clean(df):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ<>')
    text_features = df[['name',  'description', 'neighborhood_overview', 'notes', 
    'transit', 'access', 'interaction', 'house_rules', 'host_about']]\
        .fillna('<unk>').apply(lambda se: se.apply(lambda s: s.replace('/', ' ')))
    text_features = text_features.apply(lambda se: se.apply(\
        lambda s: ''.join(filter(whitelist.__contains__, s))))
    return text_features
