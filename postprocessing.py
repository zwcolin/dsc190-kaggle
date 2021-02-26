import numpy as np
import pandas as pd
import os.path as osp

SERIALS_PATH = osp.join('data', 'serials.csv')
SERIALS = pd.read_csv(SERIALS_PATH)['0'].tolist()
FILES = [osp.join('data', f'{SERIAL}.csv') for SERIAL in SERIALS]

def get_price_features():
	train_df = pd.read_csv("train.csv", low_memory=False)
	test_df = pd.read_csv("test.csv", low_memory=False)
	train_size = train_df.shape[0]
	df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

	price_mask = df['price'].groupby(df['host_id']).transform('count')>=1
	temp = df['price'].groupby(df['host_id']).transform('mean')\
	*price_mask.apply(lambda x: np.nan if x == 0 else x)
	test_estimates = temp[train_size:]
	test_indices = test_estimates.index

	return test_estimates

def execute(include_price=True):
	pred = pd.read_csv(FILES[0])['Predicted'].copy()
	for i in range(1, len(FILES)):
		pred += pd.read_csv(FILES[i])['Predicted']
	pred /= len(FILES)
	submission = pd.DataFrame()
	submission['Id'] = pd.read_csv('test.csv', low_memory=False)['id']
	submission['Predicted'] = pred
	if include_price:
		estimates = pd.Series(get_price_features())
		submission['estimates'] = estimates
		submission['estimates'] = submission['estimates'].fillna(submission['Predicted'])
		submission['Predicted'] = submission[['Predicted', 'estimates']].apply(\
			lambda row: np.average(row, weights=[0.5,0.5]), axis=1)
		submission = submission.drop(['estimates'], axis=1)
	fname = 'submission.csv'
	submission.to_csv(fname, index=False)
	print(f'success - please submit {fname} for Kaggle evaluation')

	return
