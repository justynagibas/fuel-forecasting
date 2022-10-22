import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import keras.losses as loss
import keras.metrics as metrics
from python.main.models.fuel_prices_columns import diesel_cols, petrol_cols


class MLPNetwork:

    def __init__(self, number_of_inputs, activation_function='relu', number_of_outputs=1, number_of_hidden_layers=2):
        self.number_of_inputs = number_of_inputs
        self.activation_function = activation_function
        self.number_of_outputs = number_of_outputs
        self.number_of_hidden_layers = number_of_hidden_layers
        self.network_model = None

    def create_network(self, ratio=0.66) -> None:
        number_of_hidden_neurons = int((self.number_of_inputs+self.number_of_outputs)*ratio)
        mlp = Sequential()
        mlp.add(InputLayer(input_shape=(self.number_of_inputs,)))
        for layer in range(self.number_of_hidden_layers):
            mlp.add(Dense(number_of_hidden_neurons, activation=self.activation_function))
        mlp.add(Dense(self.number_of_outputs))
        self.network_model = mlp

    def train_network(self, fuel_data: pandas.DataFrame, fuel_type: str) -> None:
        self.network_model.compile(loss=loss.MeanSquaredError(),
                                   metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()])
        self.network_model.summary()
        if fuel_type == "diesel":
            y_train = fuel_data[[diesel_cols.price_col]].to_numpy()
            x_train = fuel_data.drop(columns=[diesel_cols.price_col, diesel_cols.date_col]).to_numpy()
        elif fuel_type == "petrol":
            y_train = fuel_data[[petrol_cols.price_col]].to_numpy()
            x_train = fuel_data.drop(columns=[petrol_cols.price_col, petrol_cols.date_col]).to_numpy()

        else:
            raise AttributeError("Invalid argument: Fuel type isn't recognized")
        self.network_model.fit(x=x_train, y=y_train)

    def predict(self, fuel_data: pandas.DataFrame, fuel_type: str) -> numpy.ndarray:
        if fuel_type == "diesel":
            x = fuel_data.drop(columns=[diesel_cols.price_col, diesel_cols.date_col]).to_numpy()
        elif fuel_type == "petrol":
            x = fuel_data.drop(columns=[petrol_cols.price_col, petrol_cols.date_col]).to_numpy()
        else:
            raise AttributeError("Invalid argument: Fuel type isn't recognized")
        prediction = self.network_model.predict(x=x)
        return prediction
