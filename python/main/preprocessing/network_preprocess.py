import pandas
from python.main.models.fuel_prices_columns import petrol_cols, diesel_cols


class NetworkPreprocessor:
    """
    The NetworkPreprocessor is designed for preparing fuel data for neural network prediction

    Attributes:
        max_previous_samples (int): Max number of lagged columns with fuel price

    Raises:
        AttributeError: while max_previous_samples is not an integer or while given number is negative
    """
    def __init__(self, max_previous_samples: int = 8):
        if isinstance(max_previous_samples, int) and max_previous_samples > 0:
            self.__max_number_of_previous_samples = max_previous_samples
        else:
            raise AttributeError("Invalid argument: Number of lagged data is not integer or it's negative")
        self.__names_of_columns = [petrol_cols.week_ago_1_col,
                                   petrol_cols.week_ago_2_col,
                                   petrol_cols.week_ago_3_col,
                                   petrol_cols.week_ago_4_col,
                                   petrol_cols.week_ago_5_col,
                                   petrol_cols.week_ago_6_col,
                                   petrol_cols.week_ago_7_col,
                                   petrol_cols.week_ago_8_col]

    def add_features(self, dataframe: pandas.DataFrame, fuel_type: str) -> pandas.DataFrame:
        """
        Methods add features as columns in DataFrame with lagged fuel prices. Number of added columns is equal to
        self.__max_previous_samples

        Arguments:
            dataframe (pandas.DataFrame): DataFrame where extra features will be added
            fuel_type (str): Name of fuel type (recognized: petrol, diesel )

        Returns:
            dataframe (pandas.DataFrame): DataFrame with extra features (adding inplace)

        Raises:
            AttributeError: while no dataframe is not an instance of pandas.DataFrame
                            while fuel type in not str or name is not recognized
        """
        # TODO add error while no such column found (case when fuel_type doesn't match dataframe)
        if not isinstance(dataframe, pandas.DataFrame):
            raise AttributeError("Invalid argument: No DataFrame passed")
        if fuel_type == "petrol":
            price_name_col = petrol_cols.price_col
        elif fuel_type == "diesel":
            price_name_col = diesel_cols.price_col
        else:
            raise AttributeError("Invalid argument: Fuel type isn't recognized")
        for i in range(self.__max_number_of_previous_samples):
            dataframe[self.__names_of_columns[i]] = dataframe[price_name_col].shift(i+1)
        return dataframe



