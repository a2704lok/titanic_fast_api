from pathlib import Path
import pandas as pd
import sys
import unittest
from unittest.mock import patch
import unittest
from unittest.mock import patch, MagicMock

codesetup_path = Path.home()/"Desktop"/"titanic_setup"
sys.path.append(str(codesetup_path))

from enumerate import Features, Passenger, Sex
from predictor import XgboostPrediction

def test_make_feature() -> None:
    feature = Features(
        Pclass=Passenger.FIRST_CLASS.value,
        Sex=Sex.Female.value,
        Age=25,
        SibSp=2,
        Fare=24.5
    )
    xgbpred = XgboostPrediction(path = 'dummy/model_path')
    output = xgbpred.make_feature(feature)
    assert output is not None
    assert isinstance(output, pd.DataFrame) 
    assert list(output.columns) == ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']


class TestXgboostPrediction(unittest.TestCase):
    def test_predict(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        feature = Features(
            Pclass=Passenger.FIRST_CLASS.value,
            Sex=Sex.Female.value,
            Age=25,
            SibSp=2,
            Fare=24.5
        )
        xgbpred = XgboostPrediction(path = 'dummy/model_path')

        with patch('predictor.XgboostPrediction.load_model', return_value=mock_model) as mock_load_model:
                xgb_prediction = XgboostPrediction(path="dummy_path.pkl")
                result = xgb_prediction.predict(feature)
                mock_load_model.assert_called_once()
                mock_model.predict.assert_called_once()
                args, kwargs = mock_model.predict.call_args
                self.assertIsInstance(args[0], pd.DataFrame)

