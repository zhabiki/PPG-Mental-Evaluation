import numpy as np
from scipy.signal import butter, filtfilt, stft, detrend, savgol_filter
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


    def hlf_wavelet_cwt(self, signal, fs, wavelet, pph=300, f_min=0.01, f_max=1.00):
        """
        Вычисление LF и HF на основе вейвлет-преобразования данного сигнала с указанным вейвлетом.
        """

        num_freqs = int((f_max - f_min) * pph)
        freqs = np.linspace(f_min, f_max, num=num_freqs)

        # Связывание частот и масштабов CWT
        central_freq = pywt.central_frequency(wavelet)
        scales = central_freq * fs / freqs

        cwt_coefs, cwt_freqs = pywt.cwt(
            signal, scales, wavelet, sampling_period=(1 / fs)
        )
        cwt_ampls = np.abs(cwt_coefs)

        max_time = np.argmax(cwt_ampls, axis=1)
        max_vals = cwt_ampls[np.arange(len(cwt_ampls)), max_time]

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
            'data': [cwt_ampls, cwt_freqs]
        }


    def hlf_fft_cwt(self, signal, fs):
        """
        Вычисление LF и HF на основе комплексного преобразования Фурье данного сигнала.
        """

        signal_fft = np.fft.fft(signal)
        signal_freqs = np.fft.fftfreq(len(signal), d=(1 / fs))
        signal_ampls = np.abs(signal_fft)

        pozitiv = signal_freqs > 0
        signal_freqs = signal_freqs[pozitiv]
        signal_ampls = signal_ampls[pozitiv]

        # Исследуем частоты в областях расположения LF и HF,
        # сохраняем индексы максимумов амплитуды в этих областях
        lf_roi = (signal_freqs > 0.04) & (signal_freqs <= 0.15)
        lf_max_ampl = np.max(signal_ampls[lf_roi])
        lf_max_freq = signal_freqs[lf_roi][np.argmax(signal_ampls[lf_roi])]

        hf_roi = (signal_freqs > 0.15) & (signal_freqs <= 0.40)
        hf_max_ampl = np.max(signal_ampls[hf_roi])
        hf_max_freq = signal_freqs[hf_roi][np.argmax(signal_ampls[hf_roi])]

        return {
            'lf': [lf_max_ampl, lf_max_freq],
            'hf': [hf_max_ampl, hf_max_freq],
            'data': [signal_ampls, signal_freqs]
        }


    def savgol_filter(self, signal, window_length=15, polyorder=2):
        return savgol_filter(signal, window_length, polyorder)


__all__ = ["Filtering"]
