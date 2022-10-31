from collections import namedtuple

FuelPricesCols = namedtuple('FuelPricesCols', 'date_col petrol_price_col petrol_duty_rates_col petrol_vat_col '
                                              'diesel_price_col diesel_duty_rates_col diesel_vat_col  '
                                              'not_required_1 not_required_2 not_required_3')

DieselCols = namedtuple('DieselCols', 'date_col price_col  week_ago_1_col week_ago_2_col week_ago_3_col week_ago_4_col '
                                      'week_ago_5_col week_ago_6_col week_ago_7_col week_ago_8_col')

PetrolCols = namedtuple('PetrolCols', 'date_col price_col  week_ago_1_col week_ago_2_col week_ago_3_col week_ago_4_col '
                                      'week_ago_5_col week_ago_6_col week_ago_7_col week_ago_8_col')

fuel_prices_cols = FuelPricesCols("Date", "ULSPPrice", "ULSPDutyRates", "ULSPVat", "ULSDPrice", "ULSDDutyRates",
                                  "ULSDVat", "NotRequired1", "NotRequired2", "NotRequired3")

diesel_cols = DieselCols("Date", "ULSDPrice",  "1WeekAgo", "2WeekAgo", "3WeekAgo", "4WeekAgo", "5WeekAgo", "6WeekAgo",
                         "7WeekAgo", "8WeekAgo")

petrol_cols = PetrolCols("Date", "ULSPPrice", "1WeekAgo", "2WeekAgo", "3WeekAgo", "4WeekAgo", "5WeekAgo", "6WeekAgo",
                         "7WeekAgo", "8WeekAgo")
