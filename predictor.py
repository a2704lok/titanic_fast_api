import pickle
from pathlib import Path
import pandas as pd
import sys
import logging
import os

codesetup_path = Path.home()/"Desktop"/"titanic_setup"
sys.path.append(str(codesetup_path))

from exceptions import FileAbsenceError
from enumerate import Features

class XgboostPrediction:
    def __init__(self, path: str):
        self.path = path

    def load_model(self):
        if not os.path.exists(self.path):
            logging.error(f"Model file does not exist at path: {self.path}")
            raise FileAbsenceError(f"Model file not found at {self.path}")
        try:
            with open(self.path, 'rb') as file:
                model = pickle.load(file)
                if not hasattr(model, 'predict'):
                    raise Exception(f"Loaded model doesn't have 'predict' method: {type(model)}")
                return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise FileAbsenceError(f"Failed to load model: {e}")


    def make_feature(self, feature : Features) -> pd.DataFrame:
        data = {'Pclass': [feature.Pclass.value],
                'Sex' : [feature.Sex.value],
                'Age': [feature.Age],
                'SibSp': [feature.SibSp],
                'Fare':[feature.Fare]}
        df = pd.DataFrame(data)
        logging.info("features converted to df")
        logging.info(f"Feature Data: {df}")
        return df

    def predict(self,feature : Features):
        logging.info("model is tryiningt to predict")
        loaded_model = self.load_model()
        df = self.make_feature(feature)
        predictions = loaded_model.predict(df)
        return predictions
