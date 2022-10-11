import pandas

from python.main.loading.load_data import load_fuel_data
from python.main.models.fuel_prices_columns import fuel_prices_cols


if __name__ == '__main__':
    fuel_df = load_fuel_data()
    pandas.set_option('display.max_columns', None)
    print(fuel_df.head())


