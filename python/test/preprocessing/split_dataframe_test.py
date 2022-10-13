

import pytest
import pandas
import json
from python.main.preprocessing.split_data import SplitDataFrames
from pandas.testing import assert_frame_equal

with open("./python/test/preprocessing/test_cases.json", "r") as json_file:
    test_data = json_file.read()
test_cases = json.loads(test_data)

test_cases_split_petrol_and_diesel = test_cases["split_petrol_and_diesel_data"]


@pytest.mark.parametrize("dataframe, result", test_cases_split_petrol_and_diesel)
def test_split_petrol_and_diesel_data(dataframe, result):
    if result is None:
        with pytest.raises(AttributeError):
            SplitDataFrames(dataframe)
    else:
        split_dataframes = SplitDataFrames(pandas.DataFrame(dataframe))
        petrol, diesel = split_dataframes.split_petrol_and_diesel_data()
        assert_frame_equal(pandas.DataFrame(result[0]), petrol)
        assert_frame_equal(pandas.DataFrame(result[1]), diesel)
