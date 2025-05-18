import numpy as np
import os
import joblib
from model import PsychiatryDiseasesClassifier

class Inference:
    def __init__(self):
        """Импортируем модель и переводим её в тестовый режим"""
        self.ensemble = joblib.load('saves/checkpoint.joblib')
        self.ensemble.set_mode('test')

    def predict(self, parameters: np.ndarray):
        """Вычисляем вероятности"""
        probas = self.ensemble.forward(parameters, probas=True)
        return probas

def main(parameters: np.ndarray):
    """Инференс модели"""
    inference = Inference()

    probas = inference.predict(parameters)
    return probas
