import datetime

import matplotlib.pyplot as plt
import pandas
import matplotlib.dates as dates
from models.fuel_prices_columns import petrol_cols, diesel_cols


def plot_petrol_price_over_time(petrol_dataframe: pandas.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.set(title="Petrol price in pence (years 2003-2022)", ylabel="Price in pence")
    start = datetime.datetime.strptime("09/06/2003", "%d/%m/%Y")
    date_generated = [start + datetime.timedelta(days=(7*x)) for x in range(0, len(petrol_dataframe))]
    formatter = dates.DateFormatter("%Y")
    locator = dates.YearLocator()
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)
    ax.plot(date_generated, petrol_dataframe[petrol_cols.price_col])
    ax.set_ylim([0, 200])
    ax.xaxis.set_tick_params(rotation=45)
    plt.show()


def plot_diesel_price_over_time(diesel_dataframe: pandas.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.set(title="Diesel price in pence (years 2003-2022)", ylabel="Price in pence")
    start = datetime.datetime.strptime("09/06/2003", "%d/%m/%Y")
    date_generated = [start + datetime.timedelta(days=(7 * x)) for x in range(0, len(diesel_dataframe))]
    formatter = dates.DateFormatter("%Y")
    locator = dates.YearLocator()
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)
    ax.plot(date_generated, diesel_dataframe[diesel_cols.price_col])
    ax.set_ylim([0, 200])
    ax.xaxis.set_tick_params(rotation=45)
    plt.show()


