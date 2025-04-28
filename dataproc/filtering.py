import numpy as np
from scipy.signal import butter, filtfilt


class Filtering:
    def __init__(self):
        pass

    def butter_bandpass(self, signal, fs, lowcut=0.5, highcut=10.0, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(order, [low, high], btype='band')

        signal_filtered = filtfilt(b, a, signal)
        return signal_filtered


__all__ = ["Filtering"]
