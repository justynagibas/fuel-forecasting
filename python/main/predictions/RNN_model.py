import pandas
import numpy
from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras.losses as loss
import keras.metrics as metrics
from models.fuel_prices_columns import diesel_cols, petrol_cols


class LSTMNetwork:
    """
    The LSTMNetwork class is dedicated to create and train recurrent neural network using long short-term memory and
    making prediction using build network

    Attributes:
         number_of_inputs (int): number of input neurons
         activation_function (str): name of activation function to be used in final layer
         recurrent_function (str): name of activation function for recurrent neurons
         number_of_outputs (int): number of output neurons
         number_of_hidden_layers (int): number of hidden layers used in neural network
         network_model : neural network build in the process
    """

    def __init__(self, number_of_inputs, activation_function='relu', recurrent_function='sigmoid', number_of_outputs=1,
                 number_of_hidden_layers=2, logger=None):
        self.number_of_inputs = number_of_inputs
        self.activation_function = activation_function
        self.recurrent_function = recurrent_function
        self.number_of_outputs = number_of_outputs
        self.number_of_hidden_layers = number_of_hidden_layers
        self._logger = logger
        self.network_model = None

    def create_network(self, number_of_hidden_neurons):
        """
        Method to build structure of neuraln network

        Arguments:
            ratio (double): ratio of number of neurons in hidden layers to sum of number of inputs and outputs neurons

        Returns:
            It override self.network_model with build model
        """
        lstm = Sequential()
        lstm.add(LSTM(units=number_of_hidden_neurons, input_shape=(self.number_of_inputs, 1), activation=self.activation_function,
                      recurrent_activation=self.recurrent_function, return_sequences=True))
        for layer in range(self.number_of_hidden_layers-1):
            lstm.add(LSTM(units=number_of_hidden_neurons, activation=self.activation_function,
                          recurrent_activation=self.recurrent_function, return_sequences=True))
        lstm.add(LSTM(units=number_of_hidden_neurons, activation=self.activation_function))
        lstm.add(Dense(units=self.number_of_outputs))
        self.network_model = lstm
        # lstm.summary(print_fn=self._logger.info)

    def train_network(self,fuel_data: pandas.DataFrame, validation_data: pandas.DataFrame, fuel_type: str, epochs: int, optimizer) -> None:
        """
        Method to train neural network model

        Arguments:
            fuel_data (pandas.DataFrame): data used to train network
            validation_data (pandas,.DataFrame);
            fuel_type (str): Name of fuel type (recognized: petrol, diesel)
            epochs (int):
            optimizer

        Raises:
            AttributeError: while fuel type in not str or name is not recognized
        """
        self.network_model.compile(optimizer=optimizer, loss=loss.MeanSquaredError(),
                                   metrics=[metrics.MeanAbsolutePercentageError(), metrics.MeanSquaredError()])
        if fuel_type == "diesel":
            y_train = fuel_data[[diesel_cols.price_col]].to_numpy()
            x_train = fuel_data.drop(columns=[diesel_cols.price_col, diesel_cols.date_col]).to_numpy()
            y_validate = validation_data[[diesel_cols.price_col]].to_numpy()
            x_validate = validation_data.drop(columns=[diesel_cols.price_col, diesel_cols.date_col]).to_numpy()
        elif fuel_type == "petrol":
            y_train = fuel_data[[petrol_cols.price_col]].to_numpy()
            x_train = fuel_data.drop(columns=[petrol_cols.price_col, petrol_cols.date_col]).to_numpy()
            y_validate = validation_data[[petrol_cols.price_col]].to_numpy()
            x_validate = validation_data.drop(columns=[petrol_cols.price_col, petrol_cols.date_col]).to_numpy()
        else:
            raise AttributeError("Invalid argument: Fuel type isn't recognized")
        x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_validate = numpy.reshape(x_validate, (x_validate.shape[0], x_validate.shape[1], 1))
        self.network_model.fit(x=x_train, y=y_train, epochs=epochs, validation_data=(x_validate, y_validate))
        # self._logger.info(logs.history)

    def predict(self, fuel_data: pandas.DataFrame, fuel_type: str) -> numpy.ndarray:
        """
        Method to predict fuel prices based on

        Arguments:
            fuel_data (pandas.DataFrame): data used to make prediction
            fuel_type (str):  Name of fuel type (recognized: petrol, diesel )

        Returns:
            prediction (numpy.ndarray): predicted fuel prices
        """
        if fuel_type == "diesel":
            x = fuel_data.drop(columns=[diesel_cols.price_col, diesel_cols.date_col]).to_numpy()
        elif fuel_type == "petrol":
            x = fuel_data.drop(columns=[petrol_cols.price_col, petrol_cols.date_col]).to_numpy()
        else:
            raise AttributeError("Invalid argument: Fuel type isn't recognized")
        x = numpy.reshape(x, (x.shape[0], x.shape[1], 1))
        prediction = self.network_model.predict(x=x)
        return prediction
