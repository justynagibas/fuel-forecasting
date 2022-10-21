import pandas

from python.main.preprocessing.dataframe_filtrartion import DataFramePreprocessor
from python.main.preprocessing.split_data import SplitDataFrames
from python.main.preprocessing.network_preprocess import NetworkPreprocessor
from python.main.loading.load_data import LoadData
from python.main.visualization.plot_signgle_series import plot_petrol_price_over_time, plot_diesel_price_over_time
from python.main.models.fuel_prices_columns import fuel_prices_cols, petrol_cols, diesel_cols
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    load_data = LoadData()
    fuel_df = load_data.load_fuel_data()
    pandas.set_option('display.max_columns', None)
    print(fuel_df.head())
    dataframe_filter = DataFramePreprocessor(fuel_df)
    dataframe_filter.remove_not_required_columns()
    dataframe_filter.replace_nan_values()
    filtered_dataframe = dataframe_filter.get_fuel_data()
    print(filtered_dataframe.head())
    dataframe_split = SplitDataFrames(filtered_dataframe)
    petrol_dataframe, diesel_dataframe = dataframe_split.split_petrol_and_diesel_data()
    print(petrol_dataframe.head())
    print(diesel_dataframe.head())
    plot_petrol_price_over_time(petrol_dataframe=petrol_dataframe[[petrol_cols.date_col, petrol_cols.price_col]])
    plot_diesel_price_over_time(diesel_dataframe=diesel_dataframe[[diesel_cols.date_col, diesel_cols.price_col]])
    petrol_network_preprocessor = NetworkPreprocessor()
    print(petrol_network_preprocessor.add_features(dataframe=petrol_dataframe, fuel_type="petrol").head(10))

    # json_test_df = pandas.DataFrame({fuel_prices_cols.date_col: ["01/06/2021", "08/06/2021", "15/06/2021", "21/06/2021","28/06/2021"],
    #                                  fuel_prices_cols.petrol_price_col: [2.25, 1.46, 2.56, 2.34, 3.21],
    #                                  # fuel_prices_cols.diesel_price_col: [5.51, 3.21, 4.87, 3.78, 2.67],
    #                                  fuel_prices_cols.petrol_duty_rates_col: [52.22, 52.22, 52.22, 52.22, 52.22],
    #                                  # fuel_prices_cols.diesel_duty_rates_col: [52.22, 52.22, 52.22, 52.22, 52.22],
    #                                  fuel_prices_cols.petrol_vat_col: [20.0, 20.0, 20.0, 20.0, 20.0],
    #                                  # fuel_prices_cols.diesel_vat_col: [20.0, 20.0, 20.0, 20.0, 20.0],
    #                                  # petrol_cols.week_ago_1_col: [None, 2.25, 1.46, 2.56, 2.34],
    #                                  # petrol_cols.week_ago_2_col: [None, None, 2.25, 1.46, 2.56],
    #                                  # petrol_cols.week_ago_3_col: [None, None, None, 2.25, 1.46],
    #                                  # petrol_cols.week_ago_4_col: [None, None, None, None, 2.25],
    #                                  # diesel_cols.week_ago_1_col: [None, 5.51, 3.21, 4.87, 3.78],
    #                                  # diesel_cols.week_ago_2_col: [None, None, 5.51, 3.21, 4.87]
    #                                  })
    # json_test_df.to_json("example.json")




