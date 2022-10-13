import pandas

from python.main.models.fuel_prices_columns import fuel_prices_cols, diesel_cols, petrol_cols
# TODO prepare documentation


class SplitDataFrames:
    # TODO add error when no pandas.DataFrame is pass
    def __init__(self, dataframe: pandas.DataFrame):
        if isinstance(dataframe, pandas.DataFrame):
            self.__original_dataframe = dataframe
        else:
            raise AttributeError("Invalid argument")

    def split_petrol_and_diesel_data(self) -> (pandas.DataFrame, pandas.DataFrame):
        petrol_dataframe = pandas.DataFrame(data=self.__original_dataframe[[fuel_prices_cols.date_col,
                                                                            fuel_prices_cols.petrol_price_col,
                                                                            fuel_prices_cols.petrol_duty_rates_col,
                                                                            fuel_prices_cols.petrol_vat_col]],
                                            columns=[petrol_cols.date_col,
                                                     petrol_cols.price_col,
                                                     petrol_cols.duty_rates_col,
                                                     petrol_cols.vat_col])

        diesel_dataframe = pandas.DataFrame(data=self.__original_dataframe[[fuel_prices_cols.date_col,
                                                                            fuel_prices_cols.diesel_price_col,
                                                                            fuel_prices_cols.diesel_duty_rates_col,
                                                                            fuel_prices_cols.diesel_vat_col]],
                                            columns=[diesel_cols.date_col,
                                                     diesel_cols.price_col,
                                                     diesel_cols.duty_rates_col,
                                                     diesel_cols.vat_col])
        return petrol_dataframe, diesel_dataframe
