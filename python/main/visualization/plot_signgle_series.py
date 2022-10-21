import matplotlib.pyplot as plt
import pandas
import matplotlib.dates as dates
from python.main.models.fuel_prices_columns import petrol_cols, diesel_cols


def plot_petrol_price_over_time(petrol_dataframe: pandas.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(petrol_dataframe[petrol_cols.date_col], petrol_dataframe[petrol_cols.price_col])
    ax.set(title="Petrol price in pence (years 2003-2022)", xlabel="Date", ylabel="Petrol price in pence")
    ax.xaxis.set_major_locator(dates.MonthLocator(interval=2))
    ax.xaxis.set_minor_locator(dates.MonthLocator())
    plt.xticks(rotation=25)
    ax.set_ylim([0, 200])
    plt.show()


def plot_diesel_price_over_time(diesel_dataframe: pandas.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(diesel_dataframe[diesel_cols.date_col], diesel_dataframe[diesel_cols.price_col])
    ax.set(title="Diesel price in pence (years 2003-2022)", xlabel="Date", ylabel="Diesel price in pence")
    ax.xaxis.set_major_locator(dates.MonthLocator(interval=2))
    ax.xaxis.set_minor_locator(dates.MonthLocator())
    plt.xticks(rotation=25)
    ax.set_ylim([0, 200])
    plt.show()


