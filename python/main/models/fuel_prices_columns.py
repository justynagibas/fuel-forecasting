from collections import namedtuple

FuelPricesCols = namedtuple('FuelPricesCols', 'date_col petrol_price_col petrol_duty_rates_col petrol_vat_col '
                                              'diesel_price_col diesel_duty_rates_col diesel_vat_col  '
                                              'not_required_1 not_required_2 not_required_3')

DieselCols = namedtuple('DieselCols', 'date_col price_col duty_rates vat week_ago_1 week_ago_2 week_ago_3 week_ago_4 '
                                      'week_ago_6 week_ago_7 week_ago_8')

PetrolCols = namedtuple('PetrolCols', 'date_col price_col duty_rates vat week_ago_1 week_ago_2 week_ago_3 week_ago_4 '
                                      'week_ago_6 week_ago_7 week_ago_8')

fuel_prices_cols = FuelPricesCols("Date", "ULSPPrice", "ULSPDutyRates", "ULSPVat", "ULSDPrice", "ULSDDutyRates",
                                  "ULSDVat", "NotRequired1", "NotRequired2", "NotRequired3")

# diesel_cols = DieselCols("Date",  "ULSDPrice", "ULSDDutyRates", "ULSDVat",)
#
# petrol_cols = PetrolCols("Date", "ULSPPrice", "ULSPDutyRates", "ULSPVat")

