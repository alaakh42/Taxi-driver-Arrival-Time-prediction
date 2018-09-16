import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import datetime as dt
from data_munging import Config


def main():
    # create instance of config
    config = Config()
    filename_train, filename_test = "../data/train.csv", "../data/test.csv"  

    # create datasets
    train, test  = config.load_data(filename_train, filename_test)

    print("===================== LETS DO SOME PLOTTING =====================")

    # plot a normalized deestribution of trip_duration
    train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)
    plt.hist(train['log_trip_duration'].values, bins=100)
    plt.title('Distribution of the trip duration')
    plt.ylabel('number of training examples')
    plt.xlabel("trip duration(log)")
    plt.show()
    plt.savefig('../plots/Distributionofthe_tripduration.png')

	# plot the overlab of the training data and test data as you don't want them to be too similar
    N = 10000
    city_long_border = (-74, -74)
    city_lat_border = (40, 40)
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax[0].scatter(train['pickup_longitude'].values[:N], train['pickup_latitude'].values[:N], color='g', label='train', s=1, alpha=1)
    ax[1].scatter(test['pickup_longitude'].values[:N], test['pickup_latitude'].values[:N], color='b', label='test', s=1, alpha=1)
    plt.title("Overlab of the training data and test data")
    plt.show()
    plt.savefig("../plots/Overlab_trainingdata & testdata in Pickup Coordinates.png")

	## Convert datetime features
    train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
    test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
    train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)
    train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
    test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
    train['dropoff_date'] = train.dropoff_datetime.dt.date

    plt.plot(train.groupby('pickup_date').count()[['id']],'-', label='train')
    plt.plot(test.groupby('pickup_date').count()[['id']],'-', label='test')
    plt.title('Train and test period overlap')
    plt.ylabel('number of records')
    plt.show()
    plt.savefig("../plots/Overlab_trainingdata & testdata.png")

	#plot dist of trip_duration(dependent variable)
    fig, ax = plt.subplots(ncols=1, nrows=1)
    sns.distplot(train['trip_duration']/3600,ax=ax,bins=100,kde=False,hist_kws={'log':True})
    plt.title("Distribution of trip_duration(dependent variable)")
    plt.show()
    plt.savefig("../plots/Distribution_of_trip_duration(dependent variable).png")

    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.set_xlim(0, 30) # zoom in trip_duration 0-30
    sns.distplot(train['trip_duration']/3600,ax=ax,bins=1000,kde=False,hist_kws={'log':True})
    plt.title("Zoom in Trip Duration from 0-30")
    plt.show()
    plt.savefig("../plots/Zoom_in_TripDuration_from_0-30.png")

    # plot the pickup and dropoff locations
    fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))
    plt.ylim(40.6, 40.9)
    plt.xlim(-74.1,-73.7)
    ax.scatter(train['pickup_longitude'],train['pickup_latitude'], s=0.01, alpha=1)
    plt.title("Pickup and Dropoff Locations")
    plt.show()
    plt.savefig("../plots/Pickup_and_Dropoff_Locations aka (New york city map).png")


if __name__ == "__main__":
    main()


