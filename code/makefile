run:
	python3 plot.py
	python3 train.py

data:
	kaggle competitions download -c nyc-taxi-trip-duration
	if 
		kaggle datasets download -d oscarleo/new-york-city-taxi-with-osrm -f fastest_routes_test.csv 
		kaggle datasets download -d oscarleo/new-york-city-taxi-with-osrm -f fastest_routes_train_part_1.csv
		kaggle datasets download -d oscarleo/new-york-city-taxi-with-osrm -f fastest_routes_train_part_2.csv
	else
		kaggle competitions download new-york-city-taxi-with-osrm -f fastest_routes_train_part_1.csv fastest_routes_train_part_2.csv fastest_routes_test.csv
	kaggle datasets download -d mathijs/weather-data-in-new-york-city-2016


