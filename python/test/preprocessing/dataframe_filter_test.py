import pytest
import pandas
import json
from python.main.preprocessing.dataframe_filtrartion import DataFrameFiltration
from pandas.testing import assert_frame_equal

with open("./python/test/preprocessing/test_cases.json", "r") as json_file:
    test_data = json_file.read()
test_cases = json.loads(test_data)

test_case_get_fuel_data = test_cases["get_fuel_data"]
test_case_remove_not_required_columns = test_cases["remove_not_required_columns"]
test_case_replace_nan_values = test_cases["replace_nan_values"]


@pytest.mark.parametrize("dataframe,result", test_case_get_fuel_data)
def test_get_fuel_data(dataframe, result):
    if result is None:
        dataframe_filter = DataFrameFiltration(dataframe)
        assert dataframe_filter.get_fuel_data() is None, \
            "Expected result is {0}, actual result is {1}. Getting wrong class field or initialization error" \
            .format(result, dataframe_filter.get_fuel_data())
    else:
        dataframe_filter = DataFrameFiltration(pandas.DataFrame(dataframe))
        assert_frame_equal(pandas.DataFrame(result), dataframe_filter.get_fuel_data())


@pytest.mark.parametrize("dataframe,result",test_case_remove_not_required_columns)
def test_remove_not_required_columns(dataframe, result):
    dataframe_filter = DataFrameFiltration(pandas.DataFrame(dataframe))
    dataframe_filter.remove_not_required_columns()
    assert_frame_equal(dataframe_filter.get_fuel_data(), pandas.DataFrame(result))


@pytest.mark.parametrize("dataframe,result", test_case_replace_nan_values)
def test_replace_nan_values(dataframe, result):
    dataframe_filter = DataFrameFiltration(pandas.DataFrame(dataframe))
    dataframe_filter.replace_nan_values()
    assert_frame_equal(dataframe_filter.get_fuel_data(), pandas.DataFrame(result))
