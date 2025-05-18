import numpy as np
import os
import joblib

from .model import PsychiatryDiseasesClassifier


class Inference:
    def __init__(self, path: str):
        """Импортируем модель и переводим её в тестовый режим"""
        self.ensemble = joblib.load(path)
        self.ensemble.set_mode('test')

    def predict(self, parameters: np.ndarray):
        """Вычисляем вероятности"""
        probas = self.ensemble.forward(parameters, probas=True)
        return probas


# # Пример инференса модели
# inference = Inference('saves/checkpoint.joblib')
# parameters = None # <-- Многомерный массив (sdnn, lf/hf...)
# probas = inference.predict(parameters)


__all__ = ["Inference"]
