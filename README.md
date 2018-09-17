# New York City Taxi Trip Duration Prediction

This repo is an attemt to solve the New York taxi arrival time prediction kaggle competition

You can go [here](https://www.kaggle.com/c/nyc-taxi-trip-duration) to check the given training and test data to get a clear idea of the probelm in hand



## Getting Started 

1. Download the training/test data plus some outsource data using kaggle API 
	P.S. Check Kaggle API documentation and installation [link](https://github.com/Kaggle/kaggle-api) to get all requirements where it needs to be so you can successfully download data

```
make data
```

2. Load Data and do some EDA, Build the training data, train and evaluate the model with

```
make run
```


## Details

Here is the breakdown of the commands executed in `make run`:

1. Load the data, print out some statistics and do some plots on the dependent variable distribution, drop off and pick up coordinates and compare the locations in both the traiing and test data sets

```
python3 plot.py
```

2. Extract some features from the original training set like the distance, date, month,..etc. Also merges some outsource data then trains a punch of different regression models 

```
python3 train.py
```


## Project WD Components

1. code
	-  `data_utils.py`: where I load the data then do some simple data statistics
	-  `feature_extraction.py`: where I do feature extraction from original data while using some outsource data as well
	-  `plot.py`: where I do some EDA while plotting data
	-  `train.py`: where I fit and evaluate my models

2. plots
	- Contains the generated plots by the `plot.py` script

3. results
	- Should contains the `trip_duration` predictions for the test data for each f my models

4. model 
	- contains the model that acheived the lowest RMSE which is the `XGBoost` model

5. data 
	- Contains the main `train.csv` and `test.csv` data files from [The New York Taxi Competition](https://www.kaggle.com/c/nyc-taxi-trip-duration)
	- It also contains another folder `/outsource_data` which should has the outsource data which will be downloaded by running `make data` 
