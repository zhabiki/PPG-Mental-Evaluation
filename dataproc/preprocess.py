import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import heartpy as hp
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, stft

from filtering import Filtering
filtering = Filtering()


# from preprocess import PreprocessPPG
class PreprocessPPG:
    def __init__(self):
        pass

    def find_heartcycle_dists(self, ppg, fs, vis=[]):
        """
        Нахождение расстояний между шагами сердечного цикла.\n
        Развёрнутое описание алгоритма см. в "PPG-Datasets-Exploration/MAUS.ipynb"
        """

        dists = pd.DataFrame(columns=['d1', 'd2', 'd3', 'd4'])

        diastolic, _ = find_peaks(ppg * -1, distance=fs * 0.5, height=np.percentile(ppg * -1, 40))
        systolic = []

        # Не забываем (как я) учитывать смещение между началом данных и первым найденным пиком
        start_offset = diastolic[0]

        for i in range(len(diastolic)-1):
            ppg_cycle = ppg[diastolic[i] : diastolic[i+1]]

            # systolic_main, _ = find_peaks(ppg_cycle[: int(len(ppg_cycle)/7*3)-5], prominence=5.0, height=np.percentile(ppg_cycle, 60), distance=fs * 0.1)
            # systolic_main = systolic_main[np.argmax(systolic_main)] if len(systolic_main) > 0 else np.argmax(ppg_cycle[: int(len(ppg_cycle)/7*3)-5])
            systolic_main_range = slice(0, int(len(ppg_cycle) * 0.42))
            systolic_main, _ = find_peaks(ppg_cycle[systolic_main_range], height=np.percentile(ppg, 60), width=5, prominence=0.5)
            if len(systolic_main) > 0:
                systolic_main = systolic_main[np.argmax(ppg_cycle[systolic_main])]
            else:
                systolic_main = np.argmax(ppg_cycle[systolic_main_range])

            # systolic_refl, _ = find_peaks(ppg_cycle[int(len(ppg_cycle)/7*3)+5 :], prominence=4.0, height=np.percentile(ppg_cycle, 60), distance=fs * 0.1)
            # systolic_refl = systolic_refl[np.argmax(systolic_refl)] if len(systolic_refl) > 0 else np.argmax(ppg_cycle[int(len(ppg_cycle)/7*3)+5 :])
            systolic_refl_range = slice(int(len(ppg_cycle) * 0.50), len(ppg_cycle))
            systolic_refl, _ = find_peaks(ppg_cycle[systolic_refl_range], height=np.percentile(ppg, 60), width=3, prominence=0.4)
            if len(systolic_refl) > 0:
                systolic_refl = systolic_refl_range.start + systolic_refl[np.argmax(ppg_cycle[systolic_refl])]
            else:
                systolic_refl = systolic_refl_range.start + np.argmax(ppg_cycle[systolic_refl_range])

            # notch_delta = int((systolic_refl - systolic_main)/3)
            # notch_range = slice(systolic_main + notch_delta, systolic_refl - notch_delta)
            notch_range = slice(
                systolic_main + int((systolic_refl - systolic_main) * 0.2),
                systolic_refl - int((systolic_refl - systolic_main) * 0.2)
            )

            # dichrotic, _ = find_peaks(ppg_cycle[notch_range.start + diastolic[i] : notch_range.stop + diastolic[i]] * -1, prominence=0.2)
            # dichrotic = dichrotic[np.argmin(ppg_cycle[notch_range][dichrotic])] if len(dichrotic) > 0 else np.argmin(ppg_cycle[notch_range])
            dichrotic, _ = find_peaks(-ppg_cycle[notch_range], width=3, prominence=0.2)
            if len(dichrotic) > 0:
                dichrotic = notch_range.start + dichrotic[np.argmin(ppg_cycle[notch_range][dichrotic])]
            else:
                dichrotic = notch_range.start + np.argmin(ppg_cycle[notch_range])

            if 'dists' in vis:
                plt.plot(ppg_cycle)
                for m in [systolic_main, systolic_refl, dichrotic]:
                    plt.plot(m, ppg_cycle[m], 'ro')
                plt.savefig('dists.png')
                plt.close() # <-- Брейкпоинт ставить сюда

            systolic.append(diastolic[i] + systolic_main)

            dists = pd.concat([dists,
                pd.DataFrame([[
                    systolic_main,
                    dichrotic - systolic_main,
                    systolic_refl - dichrotic,
                    len(ppg_cycle) - systolic_refl
                ]], columns=dists.columns)
            ], ignore_index=True)

        return dists, start_offset


    def find_rri_ibi(self, ppg, fs, vis=[]):
        """Вычисление IB- и RR-интервалов и их точек расположения на сигнале."""
        ppg_filtered = filtering.butter_bandpass(ppg, fs, 0.5, 10.0)

        r_peaks, _ = find_peaks(
            ppg_filtered,
            distance=fs * 0.5, 
            height=np.percentile(ppg_filtered, 40)
        )
        d_peaks, _ = find_peaks(
            ppg_filtered * -1,
            distance=fs * 0.5,
            height=np.percentile(ppg_filtered * -1, 40)
        )

        if 'peaks' in vis:
            plt.figure(figsize=[20, 12])
            plt.plot(ppg_filtered)
            plt.plot(r_peaks, ppg_filtered[r_peaks], 'ro')
            plt.plot(d_peaks, ppg_filtered[d_peaks], 'go')
            plt.xlim(0, fs * 100)
            plt.savefig('peaks.png')
            plt.close() # <-- Брейкпоинт ставить сюда

        rri = np.diff(r_peaks / fs)
        ibi = np.diff(d_peaks / fs)
        return r_peaks, rri, d_peaks, ibi


    def find_hrv(self, ppg, fs, vis=[]):
        """Вычисление параметров ВСР с использованием HeartPy."""
        wd, m = hp.process(np.array(ppg), sample_rate=fs)

        if 'hrv' in vis:
            hp.plotter(wd, m)
            # plt.xlim(0, (wd['hr'].shape[0] / wd['sample_rate']) / 10)
            plt.savefig('hrv.png')
            plt.close() # <-- Брейкпоинт ставить сюда

        return m


    def find_lf_hf(self, rri, ppg, fs_ppg, fs_interp=4.0, vis=[]):
        """
        Вычисление параметров LF, HF и их соотношения.\n
        Развёрнутое объяснение алгоритма см. в "PPG-Datasets-Exploration/Анализ_данных_new.ipynb"
        """

        rri_cum = np.cumsum(rri)
        rri_cum = np.insert(rri_cum, 0, 0.0)[:-1]

        # Итерполируем на равномерную временнУю ось (в мануале -- 4 Гц)
        interp_times = np.arange(0, rri_cum[-1], 1/fs_interp)
        f = interp1d(rri_cum, rri, kind='cubic', fill_value='extrapolate')
        interp_rri = f(interp_times)

        # TODO: ...


    def find_rsa(self, ppg, fs, vis=[]):
        """Вычисление параметра RSA на основе соотношения LF/HF."""
        raise NotImplementedError


    def process_data(self, ppg, fs, wsize, wstride, vis=[]):
        """
        Полная обработка данных ФПГ с использованием скользящего по пикам(!) окна.

        :param ppg: Временнóе представление данных ФПГ (алгоритм не выполняет никакой фильтрации
        сигнала самостоятельно, поэтому желательно предварительно сделать это самостоятельно).

        :param fs: Частота дискретизации данных ФПГ.

        :param wsize: Размер окна — задаётся не в мс, а в количестве сердечных циклов от впадины до впадины.

        :param wstride: Шаг окна — задаётся не в мс, а в количестве сердечных циклов от впадины до впадины.

        :param vis: Список названий методов, для которых, по ходу обработки, нужно визуализировать данные
        и затем сохранить эти визуализации. Сильно замедляет работу, использовать только для отладки!

        :return results: Объект, содержащий датафрейм `params`, содержащий параметры ВСР, LF, HF и их
        соотношение, и RSA, для каждого окна, а также `rri` и `ibi` — полные данные RR- и IB-интервалов.
        """

        # Сперва находим расстояния для всего сигнала, поскольку окна
        # задаются и применяются окном от и до диастолических пиков:
        ppg_rp, ppg_rri, ppg_dp, ppg_ibi = self.find_rri_ibi(ppg, fs, vis)

        # Теперь проходим по сигналу скользящим по началам сердечных
        # циклов окном размером в wsize с.ц. с зазором в wstride с.ц.:
        params = pd.DataFrame(columns=[])

        for i in range(0, len(ppg_dp) - wsize, wstride):
            seg = ppg[ppg_dp[i] : ppg_dp[i+wsize]]
            print(f'Окно №{i}: {ppg_dp[i]}—{ppg_dp[i+wsize]} (Р: {len(seg)}, Ш: {ppg_dp[i] - ppg_dp[i-1]})')

            seg_hrv = self.find_hrv(seg, fs, vis)
            # seg_dists = heartcycle_dists.iloc[i : i+wsize].mean() # На текущий момент больше не используется!
            # seg_lf_hf = self.find_lf_hf(ppg, ppg_rri, fs, vis)
            # seg_rsa = self.find_rsa(seg_lf_hf['lf/hf'], fs, vis)

            seg_params = {
                'bpm': seg_hrv['bpm'],
                'sdnn': seg_hrv['sdnn'],
                'sdsd': seg_hrv['sdsd'],
                'rmssd': seg_hrv['rmssd'],
                'hr_mad': seg_hrv['hr_mad'],
                'sd1/sd2': seg_hrv['sd1/sd2'],

                # 'd1_mean': seg_dists['d1'],
                # 'd2_mean': seg_dists['d2'],
                # 'd3_mean': seg_dists['d3'],
                # 'd4_mean': seg_dists['d4'],

                # 'lf': seg_lf_hf['lf'],
                # 'hf': seg_lf_hf['hf'],
                # 'lf/hf': seg_lf_hf['lf/hf'],
                # 'rsa': seg_rsa
            }

            if 'seg' in vis:
                plt.figure(figsize=(12, 8))
                plt.subplot(211)
                plt.plot(seg)
                plt.subplot(212)
                plt.text(0, 0, str(seg_params)[1:-1].replace(', ', ' \n'), fontsize=16,
                         bbox=dict(facecolor='orange', alpha=0.2, edgecolor='orange'),
                         horizontalalignment='left', verticalalignment='bottom')
                plt.tight_layout()
                plt.savefig(f'seg_{i}.png')
                plt.close() # <-- Брейкпоинт ставить сюда

            # Добавляем запись в DataFrame
            params = pd.concat([params,
                pd.DataFrame([seg_params])
            ], ignore_index=True)

        return {
            'param': params, 
            'rri': ppg_rri,
            'ibi': ppg_ibi
        }


# # Пример использования на данных MAUS
# fpath = __file__.split('/preprocess.py')[0] + '/examples/maus_006_ppg_pixart_resting.csv'
# df = pd.read_csv(fpath)
# ppg_filtered = filtering.butter_bandpass(df["Resting"].to_numpy(), fs)
# p = PreprocessPPG()

# res1 = p.process_data(ppg_filtered, 100, 10, 5)
# res2 = p.process_data(ppg_filtered, 100, 20, 1)
# res3 = p.process_data(ppg_filtered, 100, 10, 10)
# res4 = p.process_data(ppg_filtered, 100, 50, 5)
# for res in [res1, res2, res3, res4]:
#     print(res, '\n') # ПКМ --> Открыть в первичном обработчике данных


# Пример использования на самопальных данных из Ардуинки
fs = 120
ppg = []
fpath = __file__.split('/preprocess.py')[0] + '/examples/250409-Н-315-120.txt'
with open(fpath, 'r') as f:
    for line in f:
        ppg.append(float(line.strip()))

ppg_filtered = filtering.butter_bandpass(ppg[100:-100], fs)
p = PreprocessPPG()
res = p.process_data(ppg_filtered, fs, 70, 1, vis=[
    'dists',
    'peaks',
    'hrv',
    'lf_hf',
    'rsa',
    # 'seg'
])
print(res['param']) # ПКМ --> Открыть в первичном обработчике данных


__all__ = ["PreprocessPPG"]
