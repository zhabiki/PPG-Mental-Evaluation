import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks
import heartpy as hp


# from preprocess import PreprocessPPG
class PreprocessPPG:
    def __init__(self):
        pass

    def find_heartcycle_dists(self, ppg, fs, vis=False):
        """
        Нахождение расстояний между шагами сердечного цикла.\n
        Развёрнутое описание алгоритма см. в "PPG-Datasets-Exploration/MAUS.ipynb"
        """

        dists = pd.DataFrame(columns=['d1', 'd2', 'd3', 'd4', 'RR', 'IB'])
        diastolic, _ = find_peaks(ppg * -1, distance=fs * 0.5, height=np.percentile(ppg * -1, 40))
        
        # Не забываем (как я) учитывать смещение между началом данных и первым найденным пиком
        start_offset = diastolic[0]

        systolic_peaks = []
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

            if vis:
                plt.plot(ppg_cycle)
                for m in [systolic_main, systolic_refl, dichrotic]:
                    plt.plot(m, ppg_cycle[m], 'ro')
                # plt.show()
                plt.close() # <-- Брейкпоинт ставить сюда

            systolic_peaks.append(diastolic[i] + systolic_main)

            dists = pd.concat([dists,
                pd.DataFrame([[
                    systolic_main,
                    dichrotic - systolic_main,
                    systolic_refl - dichrotic,
                    len(ppg_cycle) - systolic_refl,
                    0, 0 # <-- Временные значения IBI и RRI
                ]], columns=dists.columns)
            ], ignore_index=True)

        # RR-интервал -- расстояние между систолическими пиками
        rri = np.diff(systolic_peaks)
        dists.loc[:len(rri)-1, 'RR'] = rri

        # IB-интервал -- расстояние между диастолическими пиками
        ibi = np.diff(diastolic)
        dists.loc[:len(ibi)-1, 'IB'] = ibi

        return dists, start_offset
        
    
    def find_hrv(self, ppg, fs, vis=False):
        """Вычисление параметров ВСР с использованием HeartPy."""
        wd, m = hp.process(ppg, sample_rate=fs)

        if vis:
            hp.plotter(wd, m)
            # for measure in m.keys(): print('%s: %f' %(measure, m[measure]))

            plt.xlim(0, (wd['hr'].shape[0] / wd['sample_rate']) / 10)
            # plt.show()
            plt.close() # <-- Брейкпоинт ставить сюда

        return m

    def find_lf_hf(self, rr, ppg, fs, vis=False):
        """
        Вычисление параметров LF, HF и их соотношения.\n
        Развёрнутое объяснение алгоритма см. в "PPG-Datasets-Exploration/Анализ_данных.ipynb"
        """
        raise NotImplementedError

    def find_rsa(self, ppg, fs, vis=False):
        """Вычисление параметра RSA на основе соотношения LF/HF."""
        raise NotImplementedError

    def process_data(self, ppg, fs, wsize, wstride, vis=False):
        """
        Полная обработка данных ФПГ с использованием скользящего по пикам(!) окна.
        
        :param ppg: Временнóе представление данных ФПГ (алгоритм не выполняет никакой фильтрации
        сигнала самостоятельно, поэтому желательно предварительно сделать это самостоятельно).

        :param fs: Частота дискретизации данных ФПГ.

        :param wsize: Размер окна — задаётся не в мс, а в количестве сердечных циклов от впадины до впадины.

        :param wstride: Шаг окна — задаётся не в мс, а в количестве сердечных циклов от впадины до впадины.

        :param vis: Вывод и сохранение визуализации данных в процессе обработки. Использовать только для отладки!

        :return results: Датафрейм Pandas, для каждого пройденного окна содержащий средние значения
        расстояний шагов сердечного цикла, параметры ВСР, LF, HF и их соотношение, и RSA.
        """

        # Сперва находим расстояния для всего сигнала, поскольку окна
        # задаются и применяются от и до диастолических пиков сердечных циклов:
        heartcycle_dists, start_offset = self.find_heartcycle_dists(ppg, fs, vis)

        if len(heartcycle_dists) < wsize+2:
            raise ValueError("Слишком большое окно, во всём датасете пиков меньше!")

        # Теперь из расстояний предвычислим мести диастолических пиков отн. сигнала;
        # при этом не забываем (как я) учитывать стартовое смещение данных расстояний!
        offsets = [start_offset]
        for i in range(heartcycle_dists.shape[0]):
            offsets.append(offsets[-1] + round(heartcycle_dists.iloc[i]['IB']))

        # Наконец, проходим по сигналу скользящим по пикам окном с зазором в N пиков
        results = pd.DataFrame(columns=[])

        for i in range(0, len(offsets) - wsize, wstride):
            seg = ppg[offsets[i] : offsets[i+wsize]]
            print(f'Окно №{i}: {offsets[i]}--{offsets[i+wsize]} (Р: {len(seg)}, Ш: {offsets[i] - offsets[i-1]})')

            seg_hrv = self.find_hrv(seg, fs, vis)
            seg_dists = heartcycle_dists.iloc[i : i+wsize].mean()
            # TODO:
            # seg_lf_hf = self.find_lf_hf(ppg, seg_dists['RR'] fs, vis)
            # seg_rsa = self.find_rsa(seg_lf_hf['lf/hf'], fs, vis)

            seg_results = {
                'd1_mean': seg_dists['d1'],
                'd2_mean': seg_dists['d2'],
                'd3_mean': seg_dists['d3'],
                'd4_mean': seg_dists['d4'],
                'RR_mean': seg_dists['RR'],
                'IB_mean': seg_dists['IB'],

                'bpm': seg_hrv['bpm'],
                'sdnn': seg_hrv['sdnn'],
                'sdsd': seg_hrv['sdsd'],
                'rmssd': seg_hrv['rmssd'],
                'hr_mad': seg_hrv['hr_mad'],
                'sd1/sd2': seg_hrv['sd1/sd2'],

                # TODO:
                # 'lf': seg_lf_hf['lf'],
                # 'hf': seg_lf_hf['hf'],
                # 'lf/hf': seg_lf_hf['lf/hf'],
                # 'rsa': seg_rsa
            }

            if vis:
                plt.figure(figsize=(12, 8))
                plt.subplot(211)
                plt.plot(seg)
                plt.subplot(212)
                plt.text(0, 0, str(seg_results)[1:-1].replace(', ', ' \n'), fontsize=16,
                         bbox=dict(facecolor='orange', alpha=0.2, edgecolor='orange'),
                         horizontalalignment='left', verticalalignment='bottom')
                plt.tight_layout()
                # plt.show()
                plt.savefig(f'seg_{i}.png')
                plt.close() # <-- Брейкпоинт ставить сюда

            # Добавляем запись в DataFrame
            results = pd.concat([results,
                pd.DataFrame([seg_results])
            ], ignore_index=True)

        return results


# # Пример использования на данных MAUS
# fpath = __file__.split('/preprocess.py')[0] + '/examples/maus_006_ppg_pixart_resting.csv'
# df = pd.read_csv(fpath)
# p = PreprocessPPG()

# def butter_bandpass(lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
# b, a = butter_bandpass(0.5, 10, 100)
# ppg_filtered = filtfilt(b, a, df["Resting"].to_numpy())

# res1 = p.process_data(ppg_filtered, 100, 10, 5)
# res2 = p.process_data(ppg_filtered, 100, 20, 1)
# res3 = p.process_data(ppg_filtered, 100, 10, 10)
# res4 = p.process_data(ppg_filtered, 100, 50, 5)
# for res in [res1, res2, res3, res4]:
#     print(res, '\n') # ПКМ --> Открыть в первичном обработчике данных


# Пример использования на самопальных данных из Ардуинки
# fpath = __file__.split('/preprocess.py')[0] + '/examples/250409-Н-315-120.txt'
# fs = 120
# ppg = []
# with open(fpath, 'r') as f:
#     for line in f:
#         ppg.append(float(line.strip()))

# def butter_bandpass(lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return b, a
# b, a = butter_bandpass(0.5, 10, fs)
# ppg_filtered = filtfilt(b, a, ppg)#[160:6000]

# p = PreprocessPPG()
# res = p.process_data(ppg_filtered, fs, 70, 1, vis=True) # vis=True для отладки!!!
# print(res) # ПКМ --> Открыть в первичном обработчике данных


__all__ = ["PreprocessPPG"]
