import numpy as np
import heartpy as hp
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

class PreprocessSignals:
    def __init__(self):
        pass

    def butter_bandpass_filter(self, data, lowcut, highcut, sample_rate, order=4):
        nyq = 0.5 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def preprocess_signal(self, signal, size, stride, sample_rate):
        param_lst = []
        signal = self.butter_bandpass_filter(signal, 4, 9, 240)
        for i in range(0, len(signal)-size, stride):
            k = signal[i:i+size]
            wd, m = hp.process(np.array(k), sample_rate=sample_rate)

            if np.isnan(m['ibi']) or np.isnan(m['sdnn']):
                plt.plot([_ for _ in range(i, i+size)], k)
                plt.show()

            param_lst.append((m['ibi'], m['sdnn']))

        return param_lst