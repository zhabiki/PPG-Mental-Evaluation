import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import heartpy as hp
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from filtering import Filtering
filtering = Filtering()


# from preprocess import PreprocessPPG
class PreprocessPPG:
    def __init__(self, vis=[]):
        """
        :param vis: Список названий методов, для которых, по ходу обработки, нужно визуализировать данные
        и затем сохранить эти визуализации. Сильно замедляет работу, использовать только для отладки!
        """
        self.vis = vis

    def find_heartcycle_dists(self, ppg, fs):
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

            if 'dists' in self.vis:
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


    def find_rri_ibi(self, ppg, fs):
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

        if 'peaks' in self.vis:
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


    def find_hrv(self, ppg, fs):
        """Вычисление параметров ВСР с использованием HeartPy."""
        wd, m = hp.process(np.array(ppg), sample_rate=fs)

        if 'hrv' in self.vis:
            hp.plotter(wd, m)
            # plt.xlim(0, (wd['hr'].shape[0] / wd['sample_rate']) / 10)
            plt.savefig('hrv.png')
            plt.close() # <-- Брейкпоинт ставить сюда

        return m


    def find_lf_hf(self, rri, interp_fs=4.0, detrend_l=(4,5,6), approx_lr=slice(3,5), pph=300):
        """
        Вычисление параметров LF, HF и их соотношения.\n
        Развёрнутое объяснение алгоритма см. в "PPG-Datasets-Exploration/Анализ_данных_new.ipynb"
        """

        rri_cum = np.cumsum(rri)
        rri_cum = np.insert(rri_cum, 0, 0.0)[:-1]

        # Интерполируем на равномерную временнУю ось; новую fs берём 4 Гц,
        # таким образом получаем равномерное распределение с шагом 0.25 сек.
        # Графическое представление см. в "PPG-Datasets-Exploration/
        # Анализ данных как работает.png", иллюстрации 1 (до) и 2 (после).
        interp_times = np.arange(0, rri_cum[-1], 1/interp_fs)
        f = interp1d(rri_cum, rri, kind='cubic', fill_value='extrapolate')
        interp_rri = f(interp_times)

        # Теперь, из интерполированных данных, нам нужно удалить тренд
        # (должно помочь с точностью вычисленных значений, покрмр по идее).
        # Графическое представление см. в "PPG-Datasets-Exploration/
        # Анализ данных как работает.png", иллюстрации 2 (до) и 3 (после).
        interp_rri_detrended = filtering.wavelet_delevel(
            interp_rri, detrend_l, 'db6', 8
        )

        # Теперь оставляем частоты только в примерно нужном нам диапазоне
        # (нужно 0.04-0.4, а обрезаем до 0.0625-0.5 Гц), немного усредняем
        # и выполняем "выравнивание по приближённым коэффициентам" (т.е.
        # вместо изначального сигнала оставляем только приближающий тренд).
        # Графическое представление см. в "PPG-Datasets-Exploration/
        # Анализ данных как работает.png", иллюстрации 3 (до) и 4 (после).
        interp_rri_approx = np.zeros_like(
            # (просто получаем массив нулей нужной длины,
            # уровень вейвлета здесь роли не играет)
            filtering.wavelet_delevel(
                interp_rri_detrended, (0,), 'db4', 4
            )
        )

        for level in range(approx_lr.start, approx_lr.stop+1):
            interp_rri_approx += filtering.wavelet_delevel(
                interp_rri_detrended, (0,), 'db4', level
            )

        interp_rri_approx /= ((approx_lr.stop+1) - approx_lr.start)

        bias_mean = np.mean(interp_rri[int(interp_fs*5):])

        if 'lhf_plt' in self.vis:
            plt.figure(figsize=(12, 8))
            plt.subplot(211)
            plt.plot(rri, label=f'Сырые RR-интервалы: {len(rri)}')
            plt.plot(interp_rri, label=f'RRI с интерполяцией: {len(interp_rri)}')
            plt.plot(interp_rri_detrended, label='RRI с интерп. и детрендом')
            plt.plot(interp_rri_approx, label='Выравн. по прибл. коэф.')
            plt.legend()
            plt.subplot(212)
            plt.plot(interp_times[int(interp_fs*5):],
                     interp_rri[int(interp_fs*5):])
            plt.plot(interp_times[int(interp_fs*5):int(interp_fs*60*5)],
                     bias_mean + interp_rri_approx[int(interp_fs*5):int(interp_fs*60*5)])
            plt.tight_layout()
            plt.savefig('lhf_plt.png')
            plt.close()

        # Наконец, выполняем комплексное преобразование Фурье, используя
        # вейвлет Морле, оттуда находим максиумы в областях частот LF и HF.
        cwt_res = filtering.wavelet_cwt_hlf(
            interp_rri_approx, interp_fs, 'cmor1.5-1.0', pph, 0.01, 1.00
        )

        if 'lhf_cwt' in self.vis:
            plt.figure(figsize=(8, 6))
            t = np.arange(len(interp_rri_approx)) / interp_fs
            plt.pcolormesh(t, cwt_res['data'][0], cwt_res['data'][1], shading='gouraud', cmap='plasma')
            plt.ylim(0.01, 1.00)
            plt.colorbar(label='Амплитуда')
            [plt.axhline(y=y, color='cyan', linestyle='--') for y in [0.04, 0.15, 0.40]]
            plt.axhline(y=cwt_res['lf'][1], color='yellow', linewidth=0.8)
            plt.axvline(x=cwt_res['lf'][2], color='yellow', linewidth=0.8)
            plt.axhline(y=cwt_res['hf'][1], color='tomato', linewidth=0.8)
            plt.axvline(x=cwt_res['hf'][2], color='tomato', linewidth=0.8)
            plt.text(10, 0.9, f'Макс. LF: {cwt_res["lf"][0]} @ {cwt_res["lf"][1]} Гц', c='yellow')
            plt.text(10, 0.8, f'Макс. HF: {cwt_res["hf"][0]} @ {cwt_res["hf"][1]} Гц', c='tomato')
            plt.tight_layout()
            plt.savefig('lhf_cwt.png')
            plt.close()

        return {
            'lf': cwt_res['lf'][0],
            'hf': cwt_res['hf'][0],
            'lf/hf': (cwt_res['lf'][0] / cwt_res['hf'][0]) if 
                     (cwt_res['hf'][0] > 0) and (cwt_res['lf'][0] > 0) else np.nan
        }


    def find_rsa(self, ppg, fs, lf, hf):
        """Вычисление параметра RSA на основе соотношения LF/HF."""
        rsa = np.log(hf)

        # Формула взята из этого исследования:
        # https://support.mindwaretech.com/2017/09/all-about-hrv-part-4-respiratory-sinus-arrhythmia/
        # НО!!! Это ОЧЕНЬ(!) приближённое вычисление, предназначенное для ЭКГ. Его необходимо
        # заменить более точной формулой, желательно с использованием breathing_rate из HeartPy.
        return rsa


    def process_data(self, ppg, fs, wsize, wstride):
        """
        Полная обработка данных ФПГ с использованием скользящего по пикам(!) окна.

        :param ppg: Временнóе представление данных ФПГ (алгоритм не выполняет никакой фильтрации
        сигнала самостоятельно, поэтому желательно предварительно сделать это самостоятельно).

        :param fs: Частота дискретизации данных ФПГ.

        :param wsize: Размер окна — задаётся не в мс, а в количестве сердечных циклов от впадины до впадины.

        :param wstride: Шаг окна — задаётся не в мс, а в количестве сердечных циклов от впадины до впадины.

        :return results: Объект, содержащий датафрейм `params`, содержащий параметры ВСР, LF, HF и их
        соотношение, и RSA, для каждого окна, а также `rri` и `ibi` — полные данные RR- и IB-интервалов.
        """

        # Сперва находим расстояния для всего сигнала, поскольку окна
        # задаются и применяются окном от и до диастолических пиков:
        ppg_rp, ppg_rri, ppg_dp, ppg_ibi = self.find_rri_ibi(ppg, fs)

        # Теперь проходим по сигналу скользящим по началам сердечных
        # циклов окном размером в wsize с.ц. с зазором в wstride с.ц.:
        params = pd.DataFrame(columns=[])

        for i in range(0, len(ppg_dp) - wsize, wstride):
            seg = ppg[ppg_dp[i] : ppg_dp[i+wsize]]
            print(f'Окно №{i}: {ppg_dp[i]}—{ppg_dp[i+wsize]} (Р: {len(seg)}, Ш: {ppg_dp[i] - ppg_dp[i-1]})')

            # seg_dists = heartcycle_dists.iloc[i : i+wsize].mean()

            seg_hrv = self.find_hrv(seg, fs)

            seg_rri = ppg_rri[i : i+wsize]
            seg_lf_hf = self.find_lf_hf(seg_rri)

            seg_rsa = self.find_rsa(seg, fs, seg_lf_hf['lf'], seg_lf_hf['hf'])

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

                'lf': seg_lf_hf['lf'],
                'hf': seg_lf_hf['hf'],
                'lf/hf': seg_lf_hf['lf/hf'],
                'rsa': seg_rsa
            }

            if 'seg' in self.vis:
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
# fs = 100
# fpath = __file__.split('/preprocess.py')[0] + '/examples/maus_006_ppg_pixart_resting.csv'
# df = pd.read_csv(fpath)
# ppg_filtered = filtering.butter_bandpass(df["Resting"].to_numpy(), fs)
# p = PreprocessPPG()

# res1 = p.process_data(ppg_filtered, fs, 10*fs, 1)
# res2 = p.process_data(ppg_filtered, fs, 20*fs, 10)
# res3 = p.process_data(ppg_filtered, fs, 10*fs, 10)
# res4 = p.process_data(ppg_filtered, fs, 50*fs, 5)
# for res in [res1, res2, res3, res4]:
#     print(res, '\n') # ПКМ --> Открыть в первичном обработчике данных


# Пример использования на самопальных данных из Ардуинки
fs = 120
ppg = []
fpath = __file__.split('/preprocess.py')[0] + '/examples/250409-Н-315-120.txt'
with open(fpath, 'r') as f:
    for line in f:
        ppg.append(float(line.strip()))

ppg_filtered = filtering.butter_bandpass(ppg[128:-88], fs)
p = PreprocessPPG(vis=[
    # 'dists',
    # 'peaks',
    # 'hrv',
    'lhf_plt',
    'lhf_cwt',
    # 'rsa',
    # 'seg'
])
res = p.process_data(ppg_filtered, fs, 400, 1)
print(res['param']) # ПКМ --> Открыть в первичном обработчике данных


__all__ = ["PreprocessPPG"]
