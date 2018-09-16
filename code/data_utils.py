"""
Credit for EDA code to this kaggle kernel:
https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367
"""
import numpy as np
import pandas as pd 
import gc
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)

class Config():

	def __init__(self, load=True):
		"""Load the train and test sets with some basic EDA"""
		# self.train_filename = train_filename
		# self.test_filename = test_filename

	def load_data(self, train_filename, test_filename):

		self.filename_test =  train_filename
		self.filename_train = test_filename

		# Read data
		# train_cols = ['id', 'vendor_id',	'pickup_datetime', 'dropoff_datetime', 'passenger_count',	'pickup_longitude',	'pickup_latitude',	'dropoff_longitude',	'dropoff_latitude',	'store_and_fwd_flag', 'trip_duration']
		# test_cols = ['id', 'vendor_id',	'pickup_datetime', 'passenger_count',	'pickup_longitude',	'pickup_latitude',	'dropoff_longitude',	'dropoff_latitude',	'store_and_fwd_flag']

		train = pd.read_csv(train_filename, header=0) #names=train_cols, 
		test = pd.read_csv(test_filename, header=0) #names=test_cols, 

		print("===================== LETS DO SOME EDA =====================")

		# Do some data stats
		print('We have {} training rows and {} test rows.'.format(train.shape[0], test.shape[0]))
		print('We have {} training columns and {} test columns.'.format(train.shape[1], test.shape[1]))
		print(train.head(2))
		print("============================================================")
		print(test.head(2))

		# Check for NaNs
		if train.count().min() == train.shape[0] and test.count().min() == test.shape[0]:
		    print('We do not need to worry about missing values.')
		else:
		    print('oops')
		print('The store_and_fwd_flag has only two values {}.'.format(str(set(train.store_and_fwd_flag.unique()) | set(test.store_and_fwd_flag.unique()))))
		print('The vendor_id has {}/{} distincit train/test values {}.'.format(str(len(set(train.vendor_id))) , str(len(set(test.vendor_id))), str(set(train.vendor_id.unique()) | set(test.vendor_id.unique()))))

		gc.collect()

		train.dropna(inplace=True)
		test.dropna(inplace=True)

		return train, test

	

