import pytest
import pandas
import json
from python.main.preprocessing.network_preprocess import NetworkPreprocessor
from pandas.testing import assert_frame_equal

with open("./python/test/preprocessing/test_cases.json", "r") as json_file:
    test_data = json_file.read()
test_cases = json.loads(test_data)

test_cases_add_features = test_cases["add_features"]
# TODO add test cases when dataframe doesn't contains column with fuel price

@pytest.mark.parametrize("dataframe, fuel, feature_number, result", test_cases_add_features)
def test_add_features(dataframe, fuel, feature_number, result):
    if result is None:
        with pytest.raises(AttributeError):
            network_preprocess = NetworkPreprocessor(max_previous_samples=feature_number)
            network_preprocess.add_features(dataframe=dataframe, fuel_type=fuel)
    else:
        network_preprocess = NetworkPreprocessor(max_previous_samples=feature_number)
        lagged_dataframe = network_preprocess.add_features(pandas.DataFrame(dataframe), fuel_type=fuel)
        assert_frame_equal(pandas.DataFrame(result), lagged_dataframe)



