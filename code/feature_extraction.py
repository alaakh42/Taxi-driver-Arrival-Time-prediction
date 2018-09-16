import pandas as pd
import numpy as np
import datetime as dt
from data_utils import Config

def merge_data():
	"""
	
	"""

    config = Config()
    filename_train, filename_test = "../data/train.csv", "../data/test.csv"  

    # create datasets
    train, test  = config.load_data(filename_train, filename_test)

    # 1. datetime features
	# diff between weekday and day?
	#weekday - Return the day of the week as an integer, where Monday is 0 and Sunday is 6.
	#day - Between 1 and the number of days in the given month of the given year.
	train['pickup_hour'] = train.pickup_datetime.dt.hour.astype('uint8')
	train['pickup_day'] = train.pickup_datetime.dt.day.astype('uint8')
	train['pickup_weekday'] = train.pickup_datetime.dt.weekday.astype('uint8')
	train['pickup_minute'] = train.pickup_datetime.dt.minute.astype('uint8')
	train['pickup_month'] = train.pickup_datetime.dt.month.astype('unit8')
	train['pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear
	train['pickup_weekday_hour'] = train['pickup_weekday']*24 + train['pickup_hour']

	test['pickup_hour'] = test.pickup_datetime.dt.hour.astype('uint8')
	test['pickup_day'] = test.pickup_datetime.dt.day.astype('uint8')
	test['pickup_weekday'] = test.pickup_datetime.dt.weekday.astype('uint8')
	test['pickup_minute'] = test.pickup_datetime.dt.minute.astype('uint8')
	test['pickup_month'] = test.pickup_datetime.dt.month.astype('unit8')
	test['pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear
	test['pickup_weekday_hour'] = test['pickup_weekday']*24 + test['pickup_hour']

	# 2. Location features
	def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c # AVG_EARTH_RADIUS=6367
    miles = km *  0.621371
    return miles

	# def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
	#     a = haversine_array(lat1, lng1, lat1, lng2)
	#     b = haversine_array(lat1, lng1, lat2, lng1)
	#     return a + b

	# def bearing_array(lat1, lng1, lat2, lng2):
	#     AVG_EARTH_RADIUS = 6371  # in km
	#     lng_delta_rad = np.radians(lng2 - lng1)
	#     lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
	#     y = np.sin(lng_delta_rad) * np.cos(lat2)
	#     x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
	#     return np.degrees(np.arctan2(y, x))

	train['distance'] = haversine(train.pickup_longitude, train.pickup_latitude,
	                              train.dropoff_longitude, train.dropoff_latitude)
	test['distance'] = haversine(test.pickup_longitude, test.pickup_latitude,
	                              test.dropoff_longitude, test.dropoff_latitude)


	# 3. Use outsource data
	weatherdata_filename = "data/outsource_data/weather_data_nyc_centralpark_2016.csv"
	fastestroute_data_train = "data/outsource_data/fastest_train.csv"
	fastestroute_data_test = "data/outsource_data/fastest_routes_test.csv"


	wd = pd.read_csv(weatherdata_filename, header=0)
	wd['date'] = pd.to_datetime(wd.date, format="%d-%m-%Y")
	wd['pickup_day'] = wd['date'].dt.day
	wd['snow fall'] = wd['snow fall'].replace('T', 0.05).astype(np.float32) 
	wd['precipitation'] = wd['precipitation'].replace('T', 0.05).astype(np.float32) 
	wd['snow depth'] = wd['snow depth'].replace('T', 0.05).astype(np.float32) 

	# Merge training data with weather data on pickup_day
	wd_train = pd.merge(train, wd, on='pickup_day')
	wd_train = wd_train.drop(['date','maximum temperature','minimum temperature'],axis=1)

	# Merge wd_train with fastestroute_data
	fastest = pd.read_csv(fastestroute_data_train, header=0)
	wd_train_fastest = pd.merge(wd_train, fastest, on='id', how='outer')

	gc.collect()


	print("===================== CHECK TRAINING DATA =====================")

	print(wd_train_fastest.head(2))
	print("Semi-final training data shape is: {}".format(wd_train_fastest.shape))
	print("Training data colmns: {}".format(wd_train_fastest.colmns))


	# Use the same outsource data with test set
	# merge outsource data with test data as well
	ft_test_cols = [ 'id', 'starting_street','end_street','total_distance',	'total_travel_time',
					'number_of_steps','street_for_each_step','distance_per_step','travel_time_per_step',
	                 'step_maneuvers','step_direction',	'step_location_list']
	fastest_test = pd.read_csv(fastestroute_data_test, names=ft_test_cols, header=0)

	test = pd.merge(test, fastest_test, on='id', how='outer')
	test = test.drop(['step_location_list','step_direction','step_maneuvers','travel_time_per_step','distance_per_step','street_for_each_step','number_of_steps','starting_street',
	                  'end_street'], axis=1)
	test = pd.merge(test, wd, on='pickup_day')

	print("===================== CHECK TEST DATA =====================")

	print(test.head(2))
	print("Semi-final test data shape is: {}".format(test.shape))
	print("Test data colmns: {}".format(test.colmns))



	# 4. Do more data munging
	mask = ((wd_train_fastest.trip_duration > 60) & (wd_train_fastest.distance < 0.05))
	wd_train_fastest = wd_train_fastest[~mask]
	mask = (wd_train_fastest.trip_duration < 60) 
	wd_train_fastest = wd_train_fastest[~mask]
	mask =  wd_train_fastest.trip_duration > 79200
	wd_train_fastest = wd_train_fastest[~mask]
	mask = wd_train_fastest.distance/(wd_train_fastest.trip_duration/3600) > 60
	wd_train_fastest = wd_train_fastest[~mask]
	wd_train_fastest.trip_duration = wd_train_fastest.trip_duration.astype(np.uint16)
	wd_train_fastest = wd_train_fastest[wd_train_fastest.passenger_count > 0]

	# 5. Do some data maskig based on location to create jfk and lgo features
	jfk_lon = -73.778889
	jfk_lat = 40.639722
	lga_lon = -73.872611
	lga_lat = 40.77725

	wd_train_fastest['jfk_pickup_dist'] = wd_train_fastest.apply(lambda row: haversine(jfk_lon, jfk_lat, row['pickup_longitude'],row['pickup_latitude']), axis=1)
	wd_train_fastest['lga_pickup_dist'] = wd_train_fastest.apply(lambda row: haversine(lga_lon, lga_lat, row['pickup_longitude'],row['pickup_latitude']), axis=1)
	wd_train_fastest['jfk_dropoff_dist'] = wd_train_fastest.apply(lambda row: haversine(jfk_lon, jfk_lat, row['dropoff_longitude'],row['dropoff_latitude']), axis=1)
	wd_train_fastest['lga_dropoff_dist'] = wd_train_fastest.apply(lambda row: haversine(lga_lon, lga_lat, row['dropoff_longitude'],row['dropoff_latitude']), axis=1)

	wd_train_fastest['jfk'] = ((wd_train_fastest['jfk_pickup_dist'] < 2) | (wd_train_fastest['jfk_dropoff_dist'] < 2))
	wd_train_fastest['lga'] = ((wd_train_fastest['lga_pickup_dist'] < 2) | (wd_train_fastest['lga_dropoff_dist'] < 2))
	wd_train_fastest = wd_train_fastest.drop(['jfk_pickup_dist','lga_pickup_dist','jfk_dropoff_dist','lga_dropoff_dist'],axis=1)
	wd_train_fastest['workday'] = ((wd_train_fastest['pickup_hour'] > 8) & (wd_train_fastest['pickup_hour'] < 18))


	print("===================== CHECK TRAINING DATA AGAIN =====================")

	print(wd_train_fastest.head(2))
	print("Final training data shape is: {}".format(wd_train_fastest.shape))
	print("Training data colmns: {}".format(wd_train_fastest.colmns))


	return wd_train_fastest, test





