import pandas
from datetime import datetime
from models.fuel_prices_columns import fuel_prices_cols, diesel_cols, petrol_cols
# TODO add documentation and tests


def split_train_validation_and_test(dataframe: pandas.DataFrame, date: str = "01/01/2022", train_ratio: float = 0.75) \
        -> (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame):
    if not isinstance(dataframe, pandas.DataFrame):
        raise AttributeError("Invalid argument: Data need to be a DataFrame")
    if not isinstance(date, str):
        raise AttributeError("Invalid argument: Date need to be passed as string")
    if not isinstance(train_ratio, float):
        raise AttributeError("Invalid argument: Train ratio need to be an double/float")
    if train_ratio < 0 or train_ratio > 1:
        raise AttributeError("Invalid argument: Train ratio need to be in range (0,1)")
    filter = pandas.to_datetime(dataframe[fuel_prices_cols.date_col], format="%d/%m/%Y") >= datetime.strptime(date, "%d/%m/%Y")
    test = dataframe[filter]
    number_of_records = dataframe.shape[0] - test.shape[0]
    train = dataframe.iloc[:int(number_of_records*train_ratio)]
    validation = dataframe.iloc[int(number_of_records*train_ratio):number_of_records]
    return train, validation, test


class SplitDataFrames:
    """
    The SplitDataFrames is designed to separate petrol data from diesel data

    Attributes:
        dataframe (pandas.DataFrame): DataFrame containing fuel data

    Raises:
        AttributeError: while dataframe isn't instance of pandas.DataFrame
    """
    def __init__(self, dataframe: pandas.DataFrame):
        if isinstance(dataframe, pandas.DataFrame):
            self.__original_dataframe = dataframe
        else:
            raise AttributeError("Invalid argument")

    def split_petrol_and_diesel_data(self) -> (pandas.DataFrame, pandas.DataFrame):
        """
        Method split petrol and diesel data into separated DataFrames

        Returns:
            petrol_dataframe, diesel_dataframe (pandas.DataFrame, pandas.DataFrame):
            DataFrame containing peterol data (date, price, vat, duty rates),
            DataFrame containing diesel data (date, price, vat, duty rates)
        """
        petrol_dataframe = pandas.DataFrame(data=self.__original_dataframe[[fuel_prices_cols.date_col,
                                                                            fuel_prices_cols.petrol_price_col,
                                                                            # fuel_prices_cols.petrol_duty_rates_col,
                                                                            # fuel_prices_cols.petrol_vat_col
                                                                            ]],
                                            columns=[petrol_cols.date_col,
                                                     petrol_cols.price_col,
                                                     # petrol_cols.duty_rates_col,
                                                     # petrol_cols.vat_col
                                                     ])

        diesel_dataframe = pandas.DataFrame(data=self.__original_dataframe[[fuel_prices_cols.date_col,
                                                                            fuel_prices_cols.diesel_price_col,
                                                                            # fuel_prices_cols.diesel_duty_rates_col,
                                                                            # fuel_prices_cols.diesel_vat_col
                                                                            ]],
                                            columns=[diesel_cols.date_col,
                                                     diesel_cols.price_col,
                                                     # diesel_cols.duty_rates_col,
                                                     # diesel_cols.vat_col
                                                     ])
        return petrol_dataframe, diesel_dataframe


