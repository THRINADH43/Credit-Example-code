from pureml import BasePredictor, Input, Output
import pureml


class Predictor(BasePredictor):
    label = "credit_modelexample_final_desc:main:v2"
    input = Input(type="numpy ndarray")
    output = Output(type="numpy ndarray")

    def load_models(self):
        self.model = pureml.model.fetch(self.label)

    def predict(self, data):
        predictions = self.model.predict(data)

        return predictions
