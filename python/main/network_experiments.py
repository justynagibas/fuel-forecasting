
import logging
from math import sqrt

import keras.optimizers as opt
import matplotlib.pyplot as plt
import numpy
import pandas
import tensorflow as tf

from loading.load_data import LoadData
from predictions.MFFNN_model import MLPNetwork
from predictions.RNN_model import LSTMNetwork
from preprocessing.dataframe_filtrartion import DataFramePreprocessor
from preprocessing.network_preprocess import NetworkPreprocessor
from preprocessing.split_data import SplitDataFrames, split_train_validation_and_test

from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error



def swish(x, beta=1):
    return x * sigmoid(beta * x)

def leakyReLU(x, alpha=0.01):
    return tf.where(tf.greater(x, 0),  x,  alpha*x)



def search_for_optimal_parameters_mlp(train_df, test_df, validation_df, fuel_type):
    logging.basicConfig(filename="forw_result_"+fuel_type, filemode='a', level=logging.INFO)
    logger = logging.getLogger()
    get_custom_objects().update({'swish': Activation(swish)})
    get_custom_objects().update({'leakyReLU': Activation(leakyReLU)})
    input_neurons = [2,3,4,5,6,7,8]
    activation_functions = ["leakyReLU", "swish", "relu", "elu", ]
    optimizers = [opt.SGD(learning_rate=0.05), opt.RMSprop(learning_rate=0.05), opt.Adagrad(learning_rate=0.05), opt.Adadelta(learning_rate=1.0), opt.Adam(learning_rate=0.05)]
    epochs = [1, 10, 30, 50, 70, 100]
    for input_neuron in input_neurons:
        for activation_function in activation_functions:
            hidden_neurons = [(round((sqrt(1+8*input_neuron)-1)/2), "Li"), (input_neuron-1,"Tamura"), (round(sqrt(input_neuron)),"Shibata"), (round(2*(input_neuron+1)/3),"0.66")]
            for neuron_number in hidden_neurons:
                for optimizer in optimizers:
                    for epoch in epochs:
                        input_data = train_df.iloc[:, :input_neuron+2]
                        validation = validation_df.iloc[:, :input_neuron+2]
                        test = test_df.iloc[:, :input_neuron+2]
                        params = "number_input={0}_function={1}_hidden_neurons={2}_optimizer={3}_epochs={4}".format(input_neuron,activation_function,neuron_number[1],optimizer._name, epoch)
                        logger.info("Network params"+params)
                        test_network = MLPNetwork(number_of_inputs=input_neuron, activation_function=activation_function,
                                                  number_of_outputs=1, logger=logger)
                        test_network.create_network(number_of_hidden_neurons=neuron_number[0])
                        test_network.train_network(fuel_data=input_data.dropna(axis=0), validation_data=validation,
                                                   fuel_type=fuel_type, epochs=epoch, optimizer=optimizer)
                        prediction = test_network.predict(test, fuel_type)
                        y_true = test_data.iloc[:,1].reset_index(drop=True)
                        if not numpy.isnan(prediction).any():
                            mse_test, mape_test = mean_squared_error(y_true, prediction), mean_absolute_percentage_error(y_true, prediction)
                            logger.info("Test metics result")
                            results = "MSE: {0} , MAPE: {1}".format(mse_test, mape_test)
                            logger.info(results)
                        fig, ax = plt.subplots()
                        ax.plot(prediction, '-r', label="prediction")
                        ax.plot(y_true, '-b',label="actual")
                        ax.set_title("Prediction plot")
                        ax.legend()
                        plt.savefig("../../forw_plots_"+fuel_type+"/"+params+".png")


def search_for_optimal_parameters_lstm(train_df, test_df, validation_df, fuel_type):
    logging.basicConfig(filename="rec_result_"+fuel_type, filemode='a', level=logging.INFO)
    logger = logging.getLogger()
    get_custom_objects().update({'swish': Activation(swish)})
    get_custom_objects().update({'leakyReLU': Activation(leakyReLU)})
    input_neurons = [2, 3, 4, 5, 6, 7, 8]
    activation_functions = ["leakyReLU", "swish", "relu", "elu"]
    optimizers = [opt.SGD(learning_rate=0.05), opt.RMSprop(learning_rate=0.05), opt.Adagrad(learning_rate=0.05),
                  opt.Adadelta(learning_rate=1.0), opt.Adam(learning_rate=0.05)]
    epochs = [1, 10, 30, 50, 70, 100]
    recurrent_activations = ["tanh", "sigmoid","relu"]
    for input_neuron in input_neurons:
        for activation_function in activation_functions:
            for recurrent_activation in recurrent_activations:
                hidden_neurons = [(round((sqrt(1 + 8 * input_neuron) - 1) / 2), "Li"), (input_neuron - 1, "Tamura"),
                                  (round(sqrt(input_neuron)), "Shibata"), (round(2 * (input_neuron + 1) / 3), "0.66")]
                for neuron_number in hidden_neurons:
                    for optimizer in optimizers:
                        for epoch in epochs:
                            input_data = train_df.iloc[:, :input_neuron + 2]
                            validation = validation_df.iloc[:, :input_neuron + 2]
                            test = test_df.iloc[:, :input_neuron + 2]
                            params = "number_input={0}_sct_function={1}_rec_function={2}_ratio={3}_optimizer={4}_epochs={5}".\
                                format(input_neuron, activation_function, recurrent_activation, neuron_number[1], optimizer._name, epoch)
                            logger.info("Network params"+params)
                            test_network = LSTMNetwork(number_of_inputs=input_neuron, activation_function=activation_function,
                                                       recurrent_function=recurrent_activation, number_of_outputs=1, logger=logger)
                            test_network.create_network(number_of_hidden_neurons=neuron_number[0])
                            test_network.train_network(fuel_data=input_data.dropna(axis=0), validation_data=validation,
                                                       fuel_type=fuel_type, epochs=epoch, optimizer=optimizer)
                            prediction = test_network.predict(test, fuel_type)
                            y_true = test.iloc[:,1].reset_index(drop=True)
                            if not numpy.isnan(prediction).any():
                                mse_test, mape_test = mean_squared_error(y_true,prediction),mean_absolute_percentage_error(y_true, prediction)
                                logger.info("Test metics result")
                                results = "MSE: {0} , MAPE: {1}".format(mse_test, mape_test)
                                logger.info(results)
                            fig, ax = plt.subplots()
                            ax.plot(prediction, '-r', label="prediction")
                            ax.plot(y_true, '-b', label="actual")
                            ax.set_title("Prediction plot")
                            ax.legend()
                            plt.savefig("../../rec_plots_"+fuel_type+"/"+params+".png")

load_data = LoadData()
fuel_df = load_data.load_fuel_data()
pandas.set_option('display.max_columns', None)
dataframe_filter = DataFramePreprocessor(fuel_df)
dataframe_filter.remove_not_required_columns()
dataframe_filter.replace_nan_values()
filtered_dataframe = dataframe_filter.get_fuel_data()
dataframe_split = SplitDataFrames(filtered_dataframe)
petrol_dataframe, diesel_dataframe = dataframe_split.split_petrol_and_diesel_data()
petrol_network_preprocessor = NetworkPreprocessor()
petrol_lagged_dataframe = petrol_network_preprocessor.add_features(dataframe=petrol_dataframe, fuel_type="petrol")
train_data, validation_data, test_data = split_train_validation_and_test(petrol_lagged_dataframe)
# search_for_optimal_parameters_mlp(train_df=train_data, validation_df=validation_data, test_df=test_data, fuel_type="petrol")
search_for_optimal_parameters_lstm(train_data, test_data, validation_data,  fuel_type="petrol")



