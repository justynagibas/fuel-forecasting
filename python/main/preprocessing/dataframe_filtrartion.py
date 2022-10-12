
import pandas
import numpy
from typing import Union, List
from datetime import datetime
from python.main.models.fuel_prices_columns import fuel_prices_cols

# TODO create documentation, check doc string for functions, methods and classes


class DataFrameFiltration:
    """

    """

    def __init__(self, fuel_data: pandas.DataFrame, date_format="%d/%m/%Y", samples_date_diff=7, number_of_samples=10):
        if isinstance(fuel_data, pandas.DataFrame):
            self.__fuel_data = fuel_data
        else:
            self.__fuel_data = None
        self.__columns = 1
        self.__rows = 0
        self.__date_format = date_format
        self.__samples_date_diff = samples_date_diff
        self.__number_of_samples = number_of_samples

    def remove_not_required_columns(self) -> None:
        """

        :return:
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

        :return:
        """
        return self.__fuel_data

    def replace_nan_values(self) -> None:
        """

        :return:
        """
        self.__replace_dates()
        self.__replace_fuel_price()
        self.__replace_duty_rates()
        self.__replace_vat()

    def __replace_dates(self) -> None:
        nan_date_idx = self.__find_nan_indexes(fuel_prices_cols.date_col)
        for nan_idx in nan_date_idx:
            previous_date = self.__fuel_data[fuel_prices_cols.date_col][nan_idx - 1]
            self.__fuel_data[fuel_prices_cols.date_col][nan_idx] = self.__add_days_to_date(previous_date)

    def __find_nan_indexes(self, column_name) -> Union[numpy.ndarray, List]:
        try:
            return numpy.where(self.__fuel_data[column_name].isnull())[0]
        except KeyError:
            return list()

    def __add_days_to_date(self, date) -> str:
        new_date = pandas.to_datetime(date, format=self.__date_format) + pandas.Timedelta(days=self.__samples_date_diff)
        return new_date.strftime(self.__date_format)

    def __replace_fuel_price(self) -> None:
        nan_diesel_price_idx = self.__find_nan_indexes(column_name=fuel_prices_cols.petrol_price_col)
        nan_petrol_price_idx = self.__find_nan_indexes(column_name=fuel_prices_cols.diesel_price_col)
        self.__replace_price_with_moving_average(nan_idx=nan_diesel_price_idx, column_name=fuel_prices_cols.petrol_price_col)
        self.__replace_price_with_moving_average(nan_idx=nan_petrol_price_idx, column_name=fuel_prices_cols.diesel_price_col)

    def __replace_price_with_moving_average(self, nan_idx, column_name) -> None:
        for idx in nan_idx:
            moving_avg = round(self.__fuel_data[column_name][idx - self.__number_of_samples // 2:
                                                             idx + self.__number_of_samples // 2 + 1].mean(skipna=True),

                               ndigits=2)
            self.__fuel_data[column_name][idx] = moving_avg

    def __replace_vat(self) -> None:
        vat_changes = dict([("07/03/2001", 17.5),
                            ("01/12/2008", 15.0),
                            ("01/01/2010", 17.5),
                            ("09/01/2011", 20.0)])
        nan_diesel_vat_idx = self.__find_nan_indexes(column_name=fuel_prices_cols.diesel_vat_col)
        nan_petrol_vat_idx = self.__find_nan_indexes(column_name=fuel_prices_cols.petrol_vat_col)
        self.__replace_using_fix_values(nan_idx=nan_diesel_vat_idx, column_name=fuel_prices_cols.diesel_vat_col,
                                        change_dict=vat_changes)
        self.__replace_using_fix_values(nan_idx=nan_petrol_vat_idx, column_name=fuel_prices_cols.petrol_vat_col,
                                        change_dict=vat_changes)

    def __replace_using_fix_values(self, nan_idx, column_name, change_dict):
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
        self.__replace_using_fix_values(nan_idx=nan_diesel_duty_rates_idx, column_name=fuel_prices_cols.diesel_duty_rates_col,
                                        change_dict=duty_rate_changes)
        self.__replace_using_fix_values(nan_idx=nan_petrol_duty_rates_idx, column_name=fuel_prices_cols.petrol_duty_rates_col,
                                        change_dict=duty_rate_changes)
