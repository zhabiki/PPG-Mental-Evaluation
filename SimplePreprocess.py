import numpy as np
import heartpy as hp
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

class PreprocessSignals:
    def __init__(self):
        pass

    def butter_bandpass_filter(self, data, lowcut, highcut, sample_rate, order=4):
        """Встроенный фильтр Баттерворта.
        data - сигнал
        lowcut - минимально допустимая частота
        hightcut - максимально допустимая частота
        sample_rate - частота дискретизации
        order - порядок фильтра
        
        Возвращает фильтрованный сигнал"""
        nyq = 0.5 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def preprocess_signal(self, signal, size, stride, sample_rate):
        """Извлечение признаков из сигнала скользящим окном
        signal - сам сигнал
        size - размер окна
        stride - шаг окна
        sample_rate - частота дискретизации
        
        Возвращает список извлечённых параметров"""
        param_lst = []
        #Фильтруем сигнал
        signal = self.butter_bandpass_filter(signal, 4, 9, 240)
        #Проходимся окном
        for i in range(0, len(signal)-size, stride):
            #Берём сигнал для обработки
            k = signal[i:i+size]
            wd, m = hp.process(np.array(k), sample_rate=sample_rate)

            #Иногда, если сигнал очень плохой, heartpy может не извлечь из него параметры
            #В таком случае мы должны показать, где же произошёл провал (и тонким намёком послать его)
            if np.isnan(m['ibi']) or np.isnan(m['sdnn']):
                plt.plot([_ for _ in range(i, i+size)], k)
                plt.show()
            #Успех? Добавляем параметры в список
            param_lst.append((m['ibi'], m['sdnn']))

        return param_lst
