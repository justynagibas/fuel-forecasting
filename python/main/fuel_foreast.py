import datetime

import pandas
import tensorflow as tf

from matplotlib import dates

from preprocessing.dataframe_filtrartion import DataFramePreprocessor
from preprocessing.split_data import SplitDataFrames, split_train_validation_and_test
from preprocessing.network_preprocess import NetworkPreprocessor
from loading.load_data import LoadData
from visualization.plot_signgle_series import plot_petrol_price_over_time, plot_diesel_price_over_time
from models.fuel_prices_columns import fuel_prices_cols, petrol_cols, diesel_cols
from predictions.auto_ARIMA_model import ARIMAModel
from predictions.MFFNN_model import MLPNetwork
from predictions.RNN_model import LSTMNetwork
import keras.optimizers as opt
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation


def swish(x, beta=1):
    return x * sigmoid(beta * x)

def leakyReLU(x, alpha=0.01):
    return tf.where(tf.greater(x, 0),  x,  alpha*x)


if __name__ == '__main__':

    get_custom_objects().update({'swish': Activation(swish)})
    get_custom_objects().update({'leakyReLU': Activation(leakyReLU)})

    load_data = LoadData()
    fuel_df = load_data.load_fuel_data()
    # pandas.set_option('display.max_columns', None)
    # print(fuel_df.head())
    dataframe_filter = DataFramePreprocessor(fuel_df)
    dataframe_filter.remove_not_required_columns()
    dataframe_filter.replace_nan_values()
    filtered_dataframe = dataframe_filter.get_fuel_data()
    # print(filtered_dataframe.head())
    dataframe_split = SplitDataFrames(filtered_dataframe)
    petrol_dataframe, diesel_dataframe = dataframe_split.split_petrol_and_diesel_data()
    # print(petrol_dataframe.head())
    # # print(diesel_dataframe.head())
    # # plot_petrol_price_over_time(petrol_dataframe=petrol_dataframe[[petrol_cols.date_col, petrol_cols.price_col]])
    # plot_diesel_price_over_time(diesel_dataframe=diesel_dataframe[[diesel_cols.date_col, diesel_cols.price_col]])
    petrol_network_preprocessor = NetworkPreprocessor()
    lagged_petrol = petrol_network_preprocessor.add_features(dataframe=diesel_dataframe, fuel_type="diesel")
    train_data, validation_data, test_data = split_train_validation_and_test(lagged_petrol)
    # test_network = MLPNetwork(number_of_inputs=2, number_of_outputs=1, activation_function="leakyReLU")
    #
    # input_data = train_data.iloc[:, :4]
    # validation = validation_data.iloc[:, :4]
    test = test_data.iloc[:, :4]
    #
    # test_network.create_network(number_of_hidden_neurons=1)
    # test_network.train_network(fuel_data=input_data.dropna(axis=0), validation_data=validation, fuel_type="diesel", epochs=10, optimizer=opt.Adagrad(learning_rate=0.05))
    # prediction = test_network.predict(test, "diesel")
    # print(prediction)
    #
    date_generated = [datetime.datetime.strptime("03/01/2022", "%d/%m/%Y") + datetime.timedelta(days=(7 * x)) for x in range(0, 40)]
    formatter = dates.DateFormatter("%m")
    locator = dates.MonthLocator()
    #
    # fig, ax = plt.subplots()
    # ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_major_locator(locator)
    # ax.plot(date_generated, prediction, label="prediction")
    # ax.plot(date_generated, test_data.iloc[:,1],label="actual")
    # plt.show()
    #
    #
    # input_data = train_data.iloc[:, :7]
    # validation = validation_data.iloc[:, :7]
    # test = test_data.iloc[:, :7]
    #
    # test_rec_network = LSTMNetwork(number_of_inputs=5, number_of_outputs=1,activation_function='leakyReLU', recurrent_function='tanh')
    # test_rec_network.create_network(number_of_hidden_neurons=4)
    # test_rec_network.train_network(fuel_data=input_data.dropna(axis=0), validation_data=validation, fuel_type="diesel", epochs=70, optimizer=opt.Adagrad(learning_rate=0.05))
    # prediction_rec = test_rec_network.predict(test, "diesel")
    # print((prediction_rec))

    # fig, ax = plt.subplots()
    # ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_major_locator(locator)
    # ax.plot(date_generated, prediction_rec, label="prediction")
    # ax.plot(date_generated, test_data.iloc[:,1],label="actual")
    # plt.show()

    # arima_model = ARIMAModel()
    # arima_data = pandas.concat([train_data, validation_data, test_data])
    # arima_data = arima_data[petrol_cols.price_col]
    # arima_model.train(arima_data[diesel_cols.price_col])
    # arima_model = ARIMA(arima_data, order=(8,1,1))
    test_data = test_data[diesel_cols.price_col]
    # print(arima_model.model.summary())
    # error = arima_model.validate(x_validate=test_data[diesel_cols.week_ago_1_col],y_validate=test_data[diesel_cols.price_col])
    # print(error)
    # current_year = arima_model.fit().predict(start=969, n_periods=40)

    # mse = mean_squared_error(test_data, current_year)
    # mape = mean_absolute_percentage_error(test_data, current_year)
    # print(mse,mape)
    current_year = np.array([None for i in range(40)])
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)
    ax.plot(date_generated, current_year, '-r', label="prediction")
    ax.plot(date_generated, test_data, '-b', label="actual")
    # ax.set_ylim(142,195)
    ax.set(title="Prediction using MLP for diesel price (year 2022)", xlabel="Month", ylabel="Price in pence")
    ax.legend()
    plt.show()





