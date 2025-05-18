import numpy as np
import os

from .model import PsychiatryDiseasesClassifier

class Inference:
    def __init__(self, names, disorders):
        """Импортируем модель и переводим её в тестовый режим"""
        self.ensemble = PsychiatryDiseasesClassifier(disorders)
        for d,n in zip(disorders,names):
            self.ensemble.models[d].load_model(n,format='json')
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
