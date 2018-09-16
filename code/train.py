"""
Original Data fields

id - a unique identifier for each trip
vendor_id - a code indicating the provider associated with the trip record
pickup_datetime - date and time when the meter was engaged
dropoff_datetime - date and time when the meter was disengaged
passenger_count - the number of passengers in the vehicle (driver entered value)
pickup_longitude - the longitude where the meter was engaged
pickup_latitude - the latitude where the meter was engaged
dropoff_longitude - the longitude where the meter was disengaged
dropoff_latitude - the latitude where the meter was disengaged
store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
trip_duration - duration of the trip in seconds

Disclaimer: The decision was made to not remove dropoff coordinates from the dataset order to provide an expanded set of variables to use in Kernels.

Extracted Features is in features list below

"""
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from feature_extraction import merge_data
import pandas as pd
import numpy as np 
import xgboost
import pickle
import gc


# Load data + extracted features
wd_train_fastest, test = merge_data()

features = wd_train_fastest[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
                             'dropoff_longitude', 'dropoff_latitude','pickup_hour','pickup_day', 'pickup_weekday', 'pickup_minute',
                             'pickup_hour_weekofyear', 'pickup_weekday_hour','pickup_month', 'distance', 'precipitation', 'snow fall', 'snow depth',
                             'total_distance', 'total_travel_time','jfk','lga']]
target = wd_train_fastest['trip_duration']
tfeatures = test[['vendor_id', 'passenger_count','pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                 'dropoff_latitude', 'pickup_hour','pickup_day', 'pickup_weekday', 'pickup_minute',
                 'pickup_hour_weekofyear', 'pickup_weekday_hour', 'pickup_month', 'distance',
                 'total_distance', 'total_travel_time', 'precipitation',
                 'snow fall', 'snow depth','jfk','lga']]

def rmsle(evaluator, X, real):
  """ a function that calculate RMSE to check how good is the model"""
  sum = 0.0
  predicted = evaluator.predict(X)
  print("Number predicted less than 0: {}".format(np.where(predicted < 0)[0].shape))

  predicted[predicted < 0] = 0
  for x in range(len(predicted)):
      p = np.log(predicted[x]+1)
      r = np.log(real[x]+1)
      sum = sum + (p-r)**2
  return (sum/len(predicted))**0.5

# >>>> I WILL MAKE THE ASSUMTION THAT total_travel_time from fastest_test == trip_duration 
# then check the rmse value then
benchmark = fastest_test[['id','total_travel_time']]
benchmark = benchmark.rename(index=str, columns={"total_travel_time": "trip_duration"})
benchmark['trip_duration'].isnull().values.any()
print("The score for benchmark:{}".format(mean_squared_error(target, fastest_test.trip_duration))) 
benchmark.to_csv('results/benchmark.csv', index=False)

# >>>> TRY LINEAR REGRESSION AS A BENCHMARK
reg = LinearRegression()
cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
print("The Cross-validation rmse score for Linear Regression:{}".format(cross_val_score(reg, features, np.ravel(target), cv=cv, scoring=rmsle, n_jobs=-1)))
reg.fit(features, target)
pred = reg.predict(tfeatures)
print(np.where(pred < 0)[0].shape)
pred[pred < 0]=0
test['trip_duration']=pred.astype(int)
out = test[['id','trip_duration']]
out.to_csv("../results/LinearRegression_preds.csv", index=False)

# >>>> TRY KNeighbors REGRESSION  
neigh = KNeighborsRegressor(n_neighbors=10)
cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
print("The Cross-validation rmse score for KNeighbors Regressor:{}".format(cross_val_score(neigh, features, np.ravel(target), cv=cv,scoring=rmsle, n_jobs=-1)))
neigh.fit(features, target)
pred = neigh.predict(tfeatures)
print("KNeighborsRegressor Predictions shape: {}".format(np.where(pred < 0)[0].shape))
test['trip_duration']=pred.astype(int)
out = test[['id','trip_duration']]
out.to_csv("../results/KNeighbors_preds.csv", index=False)

# >>>> TRY RANDOM FOREST REGRESSOR
rf = RandomForestRegressor()
cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
print("The Cross-validation rmse score for RandomForest Regressor:{}".format(cross_val_score(rf, features, np.ravel(target), cv=cv,scoring=rmsle, n_jobs=-1)))
rf = rf.fit(features,np.ravel(target))
pred = rf.predict(tfeatures)
print("RandomForestRegressor Predictions shape: {}".format(np.where(pred < 0)[0].shape))
test['trip_duration']=pred.astype(int)
out = test[['id','trip_duration']]
out.to_csv("../results/RandomForest_preds.csv", index=False)

# >>>> FINALLY TRY XGBOOST
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
print("The Cross-validation rmse score for XGBOOST Regressor:{}".format(cross_val_score(xgb, features, np.ravel(target), cv=cv,scoring=rmsle, n_jobs=-2)))
xgb.fit(features, target)
pred = xgb.predict(tfeatures)
pred[pred < 0] = 0
print("XGBRegressor Predictions shape: {}".format(np.where(pred < 0)[0].shape))
test['trip_duration']=pred.astype(int)
out = test[['id','trip_duration']]
out.to_csv("../results/XGBRegressor_preds.csv", index=False)
# Dump the highest score model
pickle.dump(xgb, open('../models/xgb_model.sav','wb'), protocol=2)

# plot Gradient Boosting tree 
# from xgboost import plot_tree
# plot_tree(xgb)

