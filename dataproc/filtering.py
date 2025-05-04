import numpy as np
from scipy.signal import butter, filtfilt, stft, detrend
import pywt


class Filtering:
    def __init__(self):
        pass


    def butter_bandpass(self, signal, fs, lowcut=0.5, highcut=10.0, order=4):
        """
        Создание полосового фильтра Баттерворта и применение его к сигналу.
        По сути являтся макросом для быстрого получения отфильтрованного сигнала.
        """

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(order, [low, high], btype='band')

        signal_filtered = filtfilt(b, a, signal)
        return signal_filtered


    def wavelet_delevel(self, signal, signal_levels, wavelet, wavelet_level):
        """
        Изоляция отдельных уровней частот из сигнала путём вейвлет-преобразования
        с использованием вейвлета Добеши заданного порядка и заданного уровня.
        """

        coefs = pywt.wavedec(signal, wavelet, level=wavelet_level)
        filtered_coefs = []

        # Обнуляем все уровени, кроме нужных в диапазоне коэфф-ов
        for level, freqs in enumerate(coefs):
            if level in signal_levels:
                filtered_coefs.append(freqs)
            else:
                filtered_coefs.append(np.zeros_like(freqs))

        cleaned_signal = pywt.waverec(filtered_coefs, wavelet)
        return cleaned_signal[:len(signal)] # Выравниваем по изначальной длине


    def wavelet_cwt_hlf(self, signal, fs, wavelet, pph=300, f_min=0.01, f_max=1.00):
        # Создание частотной оси
        num_freqs = int((f_max - f_min) * pph)
        freqs = np.linspace(f_min, f_max, num=num_freqs)

        # Связывание частот и масштабов CWT
        central_freq = pywt.central_frequency(wavelet)
        scales = central_freq * fs / freqs

        cwt_coefs, cwt_freqs = pywt.cwt(
            signal, scales, wavelet, sampling_period=(1/fs)
        )
        cwt_power = np.abs(cwt_coefs)

        max_time = np.argmax(cwt_power, axis=1)
        max_vals = cwt_power[np.arange(len(cwt_power)), max_time]

        # Исследуем частоты в областях расположения LF и HF,
        # сохраняем индексы максимумов амплитуды в этих областях
        lf_roi = (cwt_freqs > 0.04) & (cwt_freqs <= 0.15)
        lf_idx = np.where(lf_roi)[0][
            np.argmax(max_vals[lf_roi])
        ]

        hf_roi = (cwt_freqs > 0.15) & (cwt_freqs <= 0.40)
        hf_idx = np.where(hf_roi)[0][
            np.argmax(max_vals[hf_roi])
        ]

        lf_max_ampl = max_vals[lf_idx]
        lf_max_freq = cwt_freqs[lf_idx]
        lf_max_time = max_time[lf_idx] / fs

        hf_max_ampl = max_vals[hf_idx]
        hf_max_freq = cwt_freqs[hf_idx]
        hf_max_time = max_time[hf_idx] / fs

        return {
            'lf': [lf_max_ampl, lf_max_freq, lf_max_time],
            'hf': [hf_max_ampl, hf_max_freq, hf_max_time],
            'data': [cwt_freqs, cwt_power]
        }

__all__ = ["Filtering"]
