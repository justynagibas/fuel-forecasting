import pandas
from typing import Tuple
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


class ARIMAModel:

    def __init__(self):
        self.model = None

    def model_info(self):
        return self.model.summary()

    def train(self, y_train: pandas.DataFrame) -> None:
        self.model = auto_arima(y_train,
                                start_p=2, start_q=2, max_p=8, max_d=5, max_q=8, max_order=20, maxiter=100, seasonal=False,  trace=True,  stepwise=False)

    def validate(self, x_validate: pandas.DataFrame, y_validate: pandas.DataFrame) -> Tuple[float, float]:
        prediction_result = pandas.DataFrame(self.model.predict(n_periods=len(x_validate),X=x_validate))
        mse = mean_squared_error(y_validate, prediction_result)
        mape = mean_absolute_percentage_error(y_validate, prediction_result)
        return mse, mape

    def predict(self,start:int, n_periods: int):
        return pandas.DataFrame(self.model.predict(start=start, n_periods=n_periods))




