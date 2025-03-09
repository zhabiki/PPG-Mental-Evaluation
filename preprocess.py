import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks
import heartpy as hp


# from preprocess import PreprocessPPG
class PreprocessPPG:
    def __init__(self):
        pass

    # Развёрнутый код с комментариями см. в "PPG-Datasets-Exploration/MAUS.ipynb"
    def find_heartpy_attrs(self, ppg, fs, vis=False):
        wd, m = hp.process(ppg, sample_rate=fs)

        if vis:
            hp.plotter(wd, m)
            plt.xlim(0, (wd['hr'].shape[0] / wd['sample_rate']) / 10)
            plt.show()
            plt.close() # <-- Брейкпоинт ставить сюда

            for measure in m.keys(): print('%s: %f' %(measure, m[measure]))

        return m

    # Развёрнутый код с комментариями см. в "PPG-Datasets-Exploration/MAUS.ipynb"
    def find_heartcycle_dists(self, ppg, fs, vis=False):
        dists = pd.DataFrame(columns=['d1', 'd2', 'd3', 'd4'])
        diastolic, _ = find_peaks(ppg * -1, distance=fs * 0.5, height=np.percentile(ppg * -1, 40))
        
        # Не забываем (как я) учитывать смещение между началом данных и первым найденным пиком
        start_offset = diastolic[0]

        for i in range(len(diastolic)-1):
            # INB4: все константные значения получены методом научного тыка!
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
                plt.show()
                plt.close() # <-- Брейкпоинт ставить сюда

            dists = pd.concat([dists,
                pd.DataFrame([[
                    systolic_main,
                    dichrotic - systolic_main,
                    systolic_refl - dichrotic,
                    len(ppg_cycle) - systolic_refl
                ]], columns=dists.columns)
            ], ignore_index=True)

        return dists, start_offset
    
    def process_data(self, ppg, fs, wsize, wmargin, vis=False):
        # Сперва находим расстояния для всего сигнала, поскольку окно
        # задаётся и вычисляется от и до диастолических пиков:
        heartcycle_dists, start_offset = self.find_heartcycle_dists(ppg, fs, vis)

        if len(heartcycle_dists) < wsize+2:
            raise ValueError("Слишком большое окно, во датасете пиков меньше!")

        # Теперь из расстояний предвычислим мести диастолических пиков отн. сигнала;
        # при этом не забываем (как я) учитывать стартовое смещение данных расстояний!
        offsets = [start_offset]
        for i in range(heartcycle_dists.shape[0]):
            offsets.append(offsets[-1] + heartcycle_dists.iloc[i].sum())

        # Наконец, проходим по сигналу скользящим по пикам окном с зазором в N пиков
        results = pd.DataFrame(columns=[])

        for i in range(1, len(offsets) - wsize, wmargin):
            seg = ppg[offsets[i] : offsets[i+wsize]]

            seg_hp = self.find_heartpy_attrs(seg, fs, vis)
            seg_dists = heartcycle_dists.iloc[i : i+wsize].mean()

            # АХТУНГ -- это временный и пока НЕ завершённый вариант выходных данных!
            # В частности, вместо breathingrate будет возвращаться уже посчитанный RSA,
            # а также будут значения соотношения HF/LF на основе вейвлет анализа Вани.
            seg_results = {
                'd1_mean': seg_dists['d1'],
                'd2_mean': seg_dists['d2'],
                'd3_mean': seg_dists['d3'],
                'd4_mean': seg_dists['d4'],
                'bpm': seg_hp['bpm'],
                'ibi': seg_hp['ibi'],
                'sdnn': seg_hp['sdnn'],
                'sdsd': seg_hp['sdsd'],
                'rmssd': seg_hp['rmssd'],
                'hr_mad': seg_hp['hr_mad'],
                'sd1/sd2': seg_hp['sd1/sd2'],
                'breathingrate': seg_hp['breathingrate']
            }

            # Добавляем запись в DataFrame
            results = pd.concat([results,
                pd.DataFrame([seg_results])
            ], ignore_index=True)

        return results


# Пример использования!
# fpath = __file__.split('/preprocess.py')[0] + '/maus_006_ppg_pixart_resting.csv'
# df = pd.read_csv(fpath)
# p = PreprocessPPG()

# # Создаём и применяем полосовой фильтр
# def butter_bandpass(lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# b, a = butter_bandpass(0.5, 10, 100)
# ppg_filtered = filtfilt(b, a, df["Resting"].to_numpy())

# # vis=True должно быть ТОЛЬКО ПРИ ОТЛАДКЕ, иначе всё просто повиснет нахрен!!!
# res1 = p.process_data(ppg_filtered, 100, 10, 5)
# res2 = p.process_data(ppg_filtered, 100, 20, 1)
# res3 = p.process_data(ppg_filtered, 100, 10, 10)
# res4 = p.process_data(ppg_filtered, 100, 50, 5)

# for res in [res1, res2, res3, res4]:
#     print(res, '\n')


__all__ = ["PreprocessPPG"]
