import pandas
import os
from python.main.models.fuel_prices_columns import fuel_prices_cols


class LoadData:

    def __init__(self):
        self.__working_path = os.getcwd()
        self.__file_path = '\\data\\fuel_prices.csv'

    def __create_path(self) -> str:
        folder_path = self.__working_path.removesuffix('\\python\\main')
        file_path = folder_path+self.__file_path
        return file_path

    def load_fuel_data(self) -> pandas.DataFrame:
        data_file_path = self.__create_path()
        fuel_df = pandas.read_csv(data_file_path,
                                  encoding='unicode_escape',
                                  skiprows=3,
                                  names=[fuel_prices_cols.date_col,
                                         fuel_prices_cols.petrol_price_col,
                                         fuel_prices_cols.diesel_price_col,
                                         fuel_prices_cols.petrol_duty_rates_col,
                                         fuel_prices_cols.diesel_duty_rates_col,
                                         fuel_prices_cols.petrol_vat_col,
                                         fuel_prices_cols.diesel_vat_col,
                                         fuel_prices_cols.not_required_1,
                                         fuel_prices_cols.not_required_2,
                                         fuel_prices_cols.not_required_3])
        return fuel_df
