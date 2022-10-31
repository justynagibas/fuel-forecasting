
import logging
import keras.optimizers as opt
import matplotlib.pyplot as plt
import pandas

from loading.load_data import LoadData
from predictions.MFFNN_model import MLPNetwork
from predictions.RNN_model import LSTMNetwork
from preprocessing.dataframe_filtrartion import DataFramePreprocessor
from preprocessing.network_preprocess import NetworkPreprocessor
from preprocessing.split_data import SplitDataFrames, split_train_validation_and_test


def search_for_optimal_parameters_mlp(train_df, test_df, validation_df, fuel_type):
    logging.basicConfig(filename="forw_result_"+fuel_type, filemode='a', level=logging.INFO)
    logger = logging.getLogger()
    input_neurons = [2, 3, 4, 5, 6, 7, 8]
    activation_functions = [None, "relu"]
    ratios = [0.5, 0.6, 0.66, 0.7, 0.75]
    optimizers = [opt.Adadelta(), opt.Adagrad(), opt.Adam(), opt.Adamax(), opt.Ftrl(), opt.Nadam(), opt.RMSprop()]
    epochs = [1, 10, 30, 50, 70, 100]
    for input_neuron in input_neurons:
        for activation_function in activation_functions:
            for ratio in ratios:
                for optimizer in optimizers:
                    for epoch in epochs:
                        input_data = train_df.iloc[:, :input_neuron+3]
                        validation = validation_df.iloc[:, :input_neuron+3]
                        test = test_df.iloc[:, :-(validation_df.shape[1] - input_neuron)+2]
                        params = "number_input={0}_function={1}_ratio={2}_optimizer={3}_epochs={4}".format(input_neuron,activation_function,ratio,optimizer._name, epoch)
                        logger.info("Network params"+params)
                        test_network = MLPNetwork(number_of_inputs=input_neuron, activation_function=activation_function,
                                                  number_of_outputs=1, logger=logger)
                        test_network.create_network(ratio=ratio)
                        test_network.train_network(fuel_data=input_data.dropna(axis=0), validation_data=validation,
                                                   fuel_type=fuel_type, epochs=epoch, optimizer=optimizer)
                        prediction = test_network.predict(test, fuel_type)
                        y_true = test_data.iloc[:,1].reset_index(drop=True)
                        fig, ax = plt.subplots()
                        ax.plot(prediction, '-r', label="prediction")
                        ax.plot(y_true, '-b',label="actual")
                        ax.set_title(params)
                        ax.legend()
                        plt.savefig("../../forw_plots_"+fuel_type+"/"+params+".png")


def search_for_optimal_parameters_lstm(train_df, test_df, validation_df, fuel_type):
    logging.basicConfig(filename="rec_result_"+fuel_type, filemode='a', level=logging.INFO)
    logger = logging.getLogger()
    input_neurons = [4, 5, 6, 7, 8, 9, 10]
    activation_functions = [None, "relu"]
    recurrent_activations = [None, "than", "sigmoid","relu"]
    ratios = [0.5, 0.6, 0.66, 0.7, 0.75]
    optimizers = [opt.Adadelta(), opt.Adagrad(), opt.Adam(), opt.Adamax(), opt.Ftrl(), opt.Nadam(), opt.RMSprop(), opt.SGD()]
    epochs = [1, 5, 10, 15, 20, 25, 30]
    for input_neuron in input_neurons:
        for activation_function in activation_functions:
            for recurrent_activation in recurrent_activations:
                for ratio in ratios:
                    for optimizer in optimizers:
                        for epoch in epochs:
                            input_data = train_df.iloc[:, :10 - input_neuron]
                            validation = validation_df.iloc[:, :10 - input_neuron]
                            test = test_df.iloc[:, :10 - input_neuron]
                            params = "number_input={0}_sct_function={1}_rec_function={2}_ratio={3}_optimizer={4}_epochs={5}".\
                                format(input_neuron, activation_function, recurrent_activation, ratio, optimizer._name, epoch)
                            logger.info("Network params"+params)
                            test_network = LSTMNetwork(number_of_inputs=input_neuron, activation_function=activation_function,
                                                       recurrent_function=recurrent_activation, number_of_outputs=1, logger=logger)
                            test_network.create_network(ratio=ratio)
                            test_network.train_network(fuel_data=input_data.dropna(axis=0), validation_data=validation,
                                                       fuel_type=fuel_type, epochs=epoch, optimizer=optimizer)
                            prediction = test_network.predict(test, fuel_type)
                            y_true = test_data.iloc[:,1].reset_index(drop=True)
                            plt.plot(prediction, '-r', label="prediction")
                            plt.plot(y_true, '-b',label="actual")
                            plt.title(params)
                            plt.legend()
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
petrol_lagged_data = petrol_network_preprocessor.add_features(dataframe=petrol_dataframe, fuel_type="petrol")
train_data, validation_data, test_data = split_train_validation_and_test(petrol_lagged_data)
search_for_optimal_parameters_mlp(train_df=train_data, validation_df=validation_data, test_df=test_data, fuel_type="petrol")
# search_for_optimal_parameters_lstm(train_data, validation_data, test_data, fuel_type="petrol")



