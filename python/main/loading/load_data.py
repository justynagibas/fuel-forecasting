import pandas
import os
from python.main.models.fuel_prices_columns import fuel_prices_cols


def create_path() -> str:
    current_path = os.getcwd()
    folder_path = current_path.removesuffix('\\python\\main')
    file_path = folder_path+'\\data\\fuel_prices.csv'
    return file_path


def load_fuel_data() -> pandas.DataFrame:
    data_file_path = create_path()
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
