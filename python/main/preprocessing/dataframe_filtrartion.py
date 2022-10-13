
import pandas
import numpy
from typing import Union, List, Dict
from datetime import datetime
from python.main.models.fuel_prices_columns import fuel_prices_cols


class DataFramePreprocessor:
    """
    The DataFramePreprocessor is dedicated to remove unnecessary columns and replacing missing values

    Attributes:
        fuel_data (pandas.DataFrame): DataFrame which will be processed
        date_format (str): date format used to replace missing dates
        samples_date_diff (int): number of days between fuel samples
        number_of_samples (int):number of samples used to calculate missing prices data
    Raises:
        AttributeError: while fuel_data isn't instance of pandas.DataFrame
    """

    def __init__(self, fuel_data: pandas.DataFrame, date_format: str = "%d/%m/%Y", samples_date_diff: int = 7,
                 number_of_samples: int = 10):
        if isinstance(fuel_data, pandas.DataFrame):
            self.__fuel_data = fuel_data
        else:
            raise AttributeError("Invalid argument")
        self.__columns = 1
        self.__rows = 0
        self.__date_format = date_format
        self.__samples_date_diff = samples_date_diff
        self.__number_of_samples = number_of_samples

    def remove_not_required_columns(self) -> None:
        """
        Methods removing not required columns mark in fuel_prices_cols in python.main.models.fuel_prices_columns as not
        required. Removing columns inplace. If there isn't column that mach the names in columns_to_remove it does nothing.
        Returns:
            None
        """
        columns_to_remove = [fuel_prices_cols.not_required_1,
                             fuel_prices_cols.not_required_2,
                             fuel_prices_cols.not_required_3]
        for column in columns_to_remove:
            try:
                self.__fuel_data.drop(columns=column, axis=self.__columns, inplace=True)
            except KeyError:
                continue

    def get_fuel_data(self) -> pandas.DataFrame:
        """
        Getter for fuel_data.
        Returns:
            self.__fuel_data (pandas.DataFrame): DataFrame which is processed in class
        """
        return self.__fuel_data

    def replace_nan_values(self) -> None:
        """
        Method replacing missing values in dates, prices, vat and duty rates. It replaces values inplace. If no missing
        value is found it dose nothing.
        Returns:
            None
        """
        self.__replace_dates()
        self.__replace_fuel_price()
        self.__replace_duty_rates()
        self.__replace_vat()

    def __replace_dates(self) -> None:
        """
        Private method replacing missing dates. It replace values inplace. Missing date is calculated using
        self.__add_days_to_date
        """
        nan_date_idx = self.__find_nan_indexes(fuel_prices_cols.date_col)
        for nan_idx in nan_date_idx:
            previous_date = self.__fuel_data[fuel_prices_cols.date_col][nan_idx - 1]
            self.__fuel_data[fuel_prices_cols.date_col][nan_idx] = self.__add_days_to_date(previous_date)

    def __find_nan_indexes(self, column_name: str) -> Union[numpy.ndarray, List]:
        """
        Private method searching for indexes of missing values in column. If no such column in DataFrame it returns
        empty list.
        Params:
            columns_name (str): Name of the column in DataFrame
        Returns:
            Union[numpy.ndarray, List]: iterable container of indexes
        """
        try:
            return numpy.where(self.__fuel_data[column_name].isnull())[0]
        except KeyError:
            return list()

    def __add_days_to_date(self, date: str) -> str:
        """
        Private method to add number of days to date. Number of added dasy is equal to self.__sample_date_diff
        Params:s
             date (str): date in format as self.__date_format
        Returns:
             new_date (str): calculated date in format as self.__date_format
        """
        new_date = pandas.to_datetime(date, format=self.__date_format) + pandas.Timedelta(days=self.__samples_date_diff)
        return new_date.strftime(self.__date_format)

    def __replace_fuel_price(self) -> None:
        """
        Private method replacing missing values in columns contains fuel prices. Missing value is approximated using
        moving average. It replace values in place
        """
        nan_diesel_price_idx = self.__find_nan_indexes(column_name=fuel_prices_cols.petrol_price_col)
        nan_petrol_price_idx = self.__find_nan_indexes(column_name=fuel_prices_cols.diesel_price_col)
        self.__replace_price_with_moving_average(nan_idx=nan_diesel_price_idx, column_name=fuel_prices_cols.petrol_price_col)
        self.__replace_price_with_moving_average(nan_idx=nan_petrol_price_idx, column_name=fuel_prices_cols.diesel_price_col)

    def __replace_price_with_moving_average(self, nan_idx: Union[numpy.ndarray, List], column_name: str) -> None:
        """
        Private method replacing nan with moving average. It replaces values inplace. It calculates range of elements
        for average using self.__number_of_samples. Average have 2 digits precision.
        Params:
            nan_idx (Union[numpy.ndarray, List]): iterable container containing indexes of nan values
            column_name (str): name of column to replace nan values
        """
        for idx in nan_idx:
            moving_avg = round(self.__fuel_data[column_name][idx - self.__number_of_samples // 2:
                                                             idx + self.__number_of_samples // 2 + 1].mean(skipna=True),

                               ndigits=2)
            self.__fuel_data[column_name][idx] = moving_avg

    def __replace_vat(self) -> None:
        """
        Private method to replace missing values of vat. It replaces values inplace. Vat changes are known and are kept
        in dictionary named vat_changes. Which value is appropriated is determine by self.__fill_na_using_dict.
        """
        vat_changes = dict([("07/03/2001", 17.5),
                            ("01/12/2008", 15.0),
                            ("01/01/2010", 17.5),
                            ("09/01/2011", 20.0)])
        nan_diesel_vat_idx = self.__find_nan_indexes(column_name=fuel_prices_cols.diesel_vat_col)
        nan_petrol_vat_idx = self.__find_nan_indexes(column_name=fuel_prices_cols.petrol_vat_col)
        self.__fill_na_using_dict(nan_idx=nan_diesel_vat_idx, column_name=fuel_prices_cols.diesel_vat_col,
                                  change_dict=vat_changes)
        self.__fill_na_using_dict(nan_idx=nan_petrol_vat_idx, column_name=fuel_prices_cols.petrol_vat_col,
                                  change_dict=vat_changes)

    def __fill_na_using_dict(self, nan_idx: Union[numpy.ndarray, List], column_name: str,
                             change_dict: Dict[str, float]) -> None:
        """
        Private method to replace nan values using dictionary. It replace missing values inplace. It check in which time
        period is date of missing value and replace it using vat or duty rates value in that time.
        Params:
            nan_idx (Union[numpy.ndarray, List]): iterable container containing indexes of nan values
            column_name (str): name of column where to replace missing values
            change_dict (Dict[str, float]): dictionary containing dates and values of vat or duty rates over time
        """
        date_of_change = list(change_dict.keys())
        for idx in nan_idx:
            current_date = datetime.strptime(self.__fuel_data[fuel_prices_cols.date_col][idx], self.__date_format)
            for date_idx in range(len(date_of_change)-1):
                date = datetime.strptime(date_of_change[date_idx], self.__date_format)
                next_change_date = datetime.strptime(date_of_change[date_idx+1], self.__date_format)
                if date <= current_date <= next_change_date:
                    self.__fuel_data[column_name][idx] = change_dict[date_of_change[date_idx]]
            if pandas.isnull(self.__fuel_data[column_name][idx]):
                self.__fuel_data[column_name][idx] = change_dict[date_of_change[-1]]

    def __replace_duty_rates(self) -> None:
        """
        Private method to replace missing values of duty rates. It replaces values inplace. Duty rates changes are known
        and are kept in dictionary named duty_rates_changes. Which value is appropriated is determine by
        self.__fill_na_using_dict.
        """
        duty_rate_changes = dict([("07/03/2001", 45.82),
                                  ("01/10/2003", 47.10),
                                  ("07/12/2006", 48.35),
                                  ("01/10/2007", 50.35),
                                  ("01/12/2008", 52.35),
                                  ("01/04/2009", 54.19),
                                  ("01/09/2009", 56.19),
                                  ("01/04/2010", 57.19),
                                  ("01/10/2010", 58.19),
                                  ("01/01/2011", 58.95),
                                  ("23/03/2011", 57.95),
                                  ("23/03/2022", 52.95)])
        nan_diesel_duty_rates_idx = self.__find_nan_indexes(column_name=fuel_prices_cols.diesel_duty_rates_col)
        nan_petrol_duty_rates_idx = self.__find_nan_indexes(column_name=fuel_prices_cols.petrol_duty_rates_col)
        self.__fill_na_using_dict(nan_idx=nan_diesel_duty_rates_idx, column_name=fuel_prices_cols.diesel_duty_rates_col,
                                  change_dict=duty_rate_changes)
        self.__fill_na_using_dict(nan_idx=nan_petrol_duty_rates_idx, column_name=fuel_prices_cols.petrol_duty_rates_col,
                                  change_dict=duty_rate_changes)
