import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import heartpy as hp
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

from filtering import Filtering
filtering = Filtering()


# from preprocess import PreprocessPPG
class PreprocessPPG:
    def __init__(self, vis=[]):
        """
        :param vis: Список названий методов, для которых, по ходу обработки, нужно визуализировать данные
        и затем сохранить эти визуализации. Сильно замедляет работу, использовать только для отладки!
        """

        self.vis = set(vis)


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
                plt.savefig('vis/dists.png')
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


    def find_rri_ibi(self, ppg, fs, method, filter_order=4):
        """
        Вычисление IB- и RR-интервалов и их раположения на сигнале различными методами —
        `method='clear'` подходит для чистого сигнала ФПГ с минимальными зашумениями, а
        `method='noisy'` — если сигнал шумный и `clear` оказался слишком чувствительным.
        """

        if method == 'clear':
            ppg_filtered = filtering.butter_bandpass(ppg, fs, 0.5, 10.0, filter_order)

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

        elif method == 'noisy':
            try:
                wd, m = hp.process(np.array(ppg), sample_rate=fs)
                r_peaks = wd['peaklist']
            except Exception:
                # Усё пропало шеф! Сигнал НАСТОЛЬКО шумный, что даже HP
                # его видеть не хочет! Пытаемся выжать хотя бы что-то...
                r_peaks, _ = find_peaks(
                    ppg_filtered,
                    distance=fs * 0.5, 
                )

            # Диастолические пики HeartPy вообще не определяет, но, зная
            # расположения систолических пиков, находятся они элементарно:
            d_peaks = [r_peaks[i] + np.argmin(ppg[r_peaks[i] : r_peaks[i+1]]) for i in range(len(r_peaks)-1)]
            d_peaks = list(filter(lambda dp: ppg[dp] < np.min(ppg[r_peaks]), d_peaks))

            # И наконец, мы можем помочь HP перестать срать себе в штаны,
            # удалив дублирующиеся пики, которых, как оказалось, дофига...
            r_peaks = [d_peaks[i] + np.argmax(ppg[d_peaks[i] : d_peaks[i+1]]) for i in range(len(d_peaks)-1)]
            r_peaks = list(filter(lambda rp: ppg[rp] > np.max(ppg[d_peaks]), r_peaks))

        else:
            print(f'Метода обработки ФПГ "{method}" не существует!')
            exit(1)

        if 'peaks' in self.vis:
            plt.figure(figsize=[24, 12])
            plt.plot(ppg)
            plt.plot(r_peaks, ppg[r_peaks], 'ro')
            plt.plot(d_peaks, ppg[d_peaks], 'go')
            plt.xlim(0, fs * 100)
            plt.savefig('vis/peaks.png')
            plt.close() # <-- Брейкпоинт ставить сюда

        rri = np.diff(np.asarray(r_peaks) / fs)
        ibi = np.diff(np.asarray(d_peaks) / fs)
        return r_peaks, rri, d_peaks, ibi


    def find_hrv(self, ppg, fs):
        """Вычисление параметров ВСР с использованием HeartPy."""
        try:
            wd, m = hp.process(np.array(ppg), sample_rate=fs)

            if 'hrv' in self.vis:
                hp.plotter(wd, m)
                # plt.xlim(0, (wd['hr'].shape[0] / wd['sample_rate']) / 10)
                plt.savefig('vis/hrv.png')
                plt.close() # <-- Брейкпоинт ставить сюда

            return m

        except Exception:
            # HeartPy официально постановил, что ваш сигнал -- это кал.
            return None


    def find_lf_hf(self, rri, interp_fs=4.0, detrend_l=(4,5,6), approx_lr=slice(3,5)):
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
        interp_times = np.arange(0, rri_cum[-1], (1 / interp_fs))
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
                interp_rri_detrended, range(1, 16), 'db4', 4
            )
        )

        for level in range(approx_lr.start, approx_lr.stop+1):
            interp_rri_approx += filtering.wavelet_delevel(
                interp_rri_detrended, range(1, 16), 'db4', level
            )

        interp_rri_approx /= ((approx_lr.stop+1) - approx_lr.start)

        bias_mean = np.mean(interp_rri[int(interp_fs*5):])

        if 'lhf_plot' in self.vis:
            plt.figure(figsize=(12, 8))
            plt.subplot(211)
            plt.plot(rri, label=f'Сырые RR-интервалы: {len(rri)}')
            plt.plot(interp_rri, label=f'RRI с интерполяцией: {len(interp_rri)}')
            plt.plot(interp_rri_detrended, label='RRI с интерп. и детрендом')
            plt.plot(interp_rri_approx, label='Выравн. по прибл. коэф-ам')
            plt.legend()
            plt.subplot(212)
            plt.plot(interp_times[int(interp_fs*5):],
                     interp_rri[int(interp_fs*5):])
            plt.plot(interp_times[int(interp_fs*5):int(interp_fs*60*5)],
                     bias_mean + interp_rri_approx[int(interp_fs*5):int(interp_fs*60*5)])
            plt.tight_layout()
            plt.savefig('vis/lhf_plot.png')
            plt.close()

        # Наконец, для сигнала выполняем комплексное преобразование Фурье
        # анализируем области частот LF и HF и оттуда находим их максимумы.
        cwt_res = filtering.hlf_fft_cwt(interp_rri_approx, interp_fs)

        if 'lhf_comp' in self.vis:
            plt.figure(figsize=(12, 8))
            plt.plot(cwt_res['data'][1], cwt_res['data'][0])
            plt.xlim(0.0, 0.5)
            [plt.axvline(x=xi, color='cyan', linestyle='--') for xi in [0.04, 0.15, 0.40]]
            plt.axhline(y=cwt_res['lf'][0], color='yellow', linewidth=0.8)
            plt.axvline(x=cwt_res['lf'][1], color='yellow', linewidth=0.8)
            plt.axhline(y=cwt_res['hf'][0], color='tomato', linewidth=0.8)
            plt.axvline(x=cwt_res['hf'][1], color='tomato', linewidth=0.8)
            plt.text(0.25, cwt_res['data'][0].max() * 0.6,
                     f'Макс. LF: {cwt_res["lf"][0]} @ {cwt_res["lf"][1]} Гц', c='yellow')
            plt.text(0.25, cwt_res['data'][0].max() * 0.5,
                     f'Макс. HF: {cwt_res["hf"][0]} @ {cwt_res["hf"][1]} Гц', c='tomato')
            plt.xlabel("Частота")
            plt.ylabel("Амплитуда")
            plt.savefig('vis/lhf_comp.png')
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
    

    def remove_outliers(self, ppg, fs, peaks_pos, peaks_int, sigma_amp, sigma_int):
        """Нахождение и удаление аутлаеров, а также подозрительно длинных или коротких пиков."""
        clean_ppg = ppg.copy()
        outliers = set()
        MAD = 1.4826 # https://real-statistics.com/descriptive-statistics/mad-and-outliers/

        # Применяем правило трёх сигм для нахождения аномальных амплитуд
        amp_median = np.median(ppg[peaks_pos])
        amp_mad = MAD * np.median(np.abs(ppg[peaks_pos] - amp_median))

        amp_outliers = np.where(
            np.abs(ppg[peaks_pos] - amp_median) > (amp_mad * sigma_amp)
        )[0]

        for pos in amp_outliers:
            w_start = peaks_pos[pos-1] if pos > 0 else 0
            w_end = peaks_pos[pos+1] if pos < len(peaks_pos)-1 else len(ppg)-1
            outliers.add((w_start, w_end, 'amp'))

        # Аналогично поступаем для нахождения аномальных интервалов
        int_median = np.median(peaks_int)
        int_mad = MAD * np.median(np.abs(peaks_int - int_median))

        int_outliers = np.where(
            np.abs(peaks_int - int_median) > (int_mad * sigma_int)
        )[0]

        for pos in int_outliers:
            w_start = peaks_pos[pos-1] if pos > 0 else 0
            w_end = peaks_pos[pos+1] if pos < len(peaks_pos)-1 else len(ppg)-1
            outliers.add((w_start, w_end, 'int'))

        # Наконец, удаляем все с.ц., содержащие аутлаеры, из сигнала
        for w_start, w_end, _ in outliers:
            clean_ppg[w_start : w_end+1] = np.nan
        clean_ppg = clean_ppg[~np.isnan(clean_ppg)]

        if 'outliers' in self.vis:
            plt.figure(figsize=[24, 12])
            plt.plot(ppg)
            plt.plot(peaks_pos, ppg[peaks_pos], 'go')
            for w_start, w_end, w_reason in outliers:
                reason_color = 'orangered' if w_reason == 'amp' else 'royalblue'
                plt.axvspan(w_start, w_end, color=reason_color, alpha=0.4)
            plt.text(0, ppg.max() * 0.9, fontsize=16,
                     s=f'КРАСН — аном. амп., СИНИЙ — аном. инт.')
            plt.text(0, ppg.max() * 0.8, fontsize=16,
                     s=f'Длина до удаления аутлаеров: {len(ppg)}')
            plt.text(0, ppg.max() * 0.7, fontsize=16,
                     s=f'После удаления аутлаеров: {len(clean_ppg)} ({int((len(clean_ppg) - len(ppg)) / len(ppg) * 100)}%)')
            plt.savefig('vis/outliers.png')
            plt.close() # <-- Брейкпоинт ставить сюда

        return clean_ppg


    def process_data(self, ppg, fs, wsize, wstride, method='clear', mode='cycles'):
        """
        Полная обработка данных ФПГ с использованием скользящего по пикам(!) окна.

        :param ppg: Временнóе представление данных ФПГ (алгоритм не выполняет никакой фильтрации
        сигнала самостоятельно, поэтому желательно предварительно сделать это самостоятельно).

        :param fs: Частота дискретизации данных ФПГ.

        :param wsize: Размер окна — задаётся в количестве сердечных циклов от впадины до впадины.

        :param wstride: Шаг окна — задаётся в количестве сердечных циклов от впадины до впадины.

        :return results: Датафрейм `params`, содержащий, для каждого окна, некоторые параметры ВСР,
        усреднённые по окну IB- и RR-интервалы, LF, HF и их соотношение, а также значение RSA.
        """

        # Если метод чистый, верим юзеру на слово -- иначе, мы применяем
        # стандартный комплекс фильтрации шумов и удаления аутлайеров:
        if method != 'clear':
            ppg = filtering.butter_bandpass(ppg, fs, 0.5, 4.0, 4)
            ppg += np.abs(ppg.min())
            ppg /= np.median(ppg) # Если взять min(), то будет деление на 0!
            # ppg = (ppg - np.median(ppg)) / np.std(ppg)

            ppg_rp, ppg_rri, ppg_dp, ppg_ibi = self.find_rri_ibi(ppg, fs, method, 6)
            ppg = self.remove_outliers(ppg, fs, ppg_rp, ppg_rri, 3, 4) # <-- 3 сигмы амп., 4 сигмы инт.

            ppg = filtering.savgol_filter(ppg, 15, 2)

        params = pd.DataFrame(columns=[])


        # Сперва находим расстояния для всего сигнала, поскольку окна
        # задаются и применяются от и до диастолических пиков (aka IBI).
        # Почему именно диастолических? Потому что так графики красивше!
        # Если брать по систолическим (RRI) ничего особо не изменится.
        ppg_rp, ppg_rri, ppg_dp, ppg_ibi = self.find_rri_ibi(ppg, fs, method, 4)


        if mode == 'cycles':
            # Теперь проходим по сигналу скользящим по началам сердечных
            # циклов окном размером в wsize с.ц. с зазором в wstride с.ц.:
            for i in range(0, len(ppg_dp) - wsize, wstride):
                seg = ppg[ppg_dp[i] : ppg_dp[i+wsize]]
                print(f'Окно №{i}: {ppg_dp[i]}—{ppg_dp[i+wsize]} (≈ {int((ppg_dp[i+wsize] - ppg_dp[i]) / fs)} сек.)')
                print(f'Размер окна: {len(seg)}, Размер шага: {ppg_dp[i] - ppg_dp[i-1]}')


        elif mode == 'time':
            # Либо, в случае с режимом задания по времени, просто проходим
            # заданным в числе записей отрезом, без учёта с.ц. и прочего:
            for t in range(0, len(ppg) - wsize, wstride):
                seg = ppg[t : t+wsize]
                print(f'Отрезок на {t/fs} сек.: {t}—{t+wsize}, {(wsize / fs)} сек.)')
                print(f'Размер окна: {len(seg)}, Размер шага: {wstride}')


        else:
            print(f'Режима задания окна "{mode}" не существует!')
            exit(1)


        seg_hrv = self.find_hrv(seg, fs)

        # Иногда HeartPy может быкануть и не захотеть находить ВСР,
        # в таком случае просто пропускаем сегмент и ехаем дальше
        if seg_hrv:
            seg_rri = ppg_rri[i : i+wsize]
            seg_ibi = ppg_ibi[i : i+wsize]

            # Для корректного определения LF нужна длина минимум 25 сек,
            # на окнах меньшего размера результат не будет иметь смысла;
            # конечно, желательно, чтобы длина была хотя бы 5 минут, но
            # 25 сек -- это прям самый минимум, т.к. 1/0.04 = 25.
            if (
                (((mode == 'cycles') and (ppg_dp[i+wsize] - ppg_dp[i]) / fs) >= 25.0)
                or ((mode == 'time') and (wsize/fs) >= 25.0)
            ):
                seg_lf_hf = self.find_lf_hf(seg_rri)
                seg_rsa = self.find_rsa(seg, fs, seg_lf_hf['lf'], seg_lf_hf['hf'])
            else:
                seg_lf_hf = { 'lf': None, 'hf': None, 'lf/hf': None }
                seg_rsa = None

            seg_params = {
                'bpm': seg_hrv['bpm'],
                'sdnn': seg_hrv['sdnn'],
                'rmssd': seg_hrv['rmssd'],
                'hr_mad': seg_hrv['hr_mad'],
                'rri_mean': np.mean(seg_rri, axis=0),
                'ibi_mean': np.mean(seg_ibi, axis=0),
                'lf': seg_lf_hf['lf'],
                'hf': seg_lf_hf['hf'],
                'lf/hf': seg_lf_hf['lf/hf'],
                'rsa': seg_rsa
            }

            if 'seg' in self.vis or 'seg_i' in self.vis:
                plt.figure(figsize=(12, 8))
                plt.subplot(211)
                plt.plot(seg)
                plt.subplot(212)
                plt.text(0, 0, str(seg_params)[1:-1].replace(', ', '\n'), fontsize=16,
                        bbox=dict(facecolor='orange', alpha=0.2, edgecolor='orange'),
                        horizontalalignment='left', verticalalignment='bottom')
                plt.tight_layout()
                if 'seg_i' in self.vis:
                    plt.savefig(f'vis/seg_{i}.png')
                else:
                    plt.savefig('vis/seg.png')
                plt.close() # <-- Брейкпоинт ставить сюда

            # Добавляем запись в DataFrame
            params = pd.concat([params,
                pd.DataFrame([seg_params])
            ], ignore_index=True)

        return params


# # Пример использования на данных из датасета MAUS
# fs = 100
# fpath = __file__.split('/preprocess.py')[0] + '/examples/maus_006_ppg_pixart_resting.csv'
# df = pd.read_csv(fpath)
# ppg_filtered = filtering.butter_bandpass(df["Resting"].to_numpy(), fs)

# p = PreprocessPPG(vis=[
#     # 'dists',
#     'peaks',
#     'hrv',
#     # 'lhf_plot',
#     # 'lhf_comp',
#     # 'rsa',
#     # 'outliers',
#     'seg',
#     # 'seg_i'
# ])

# res1 = p.process_data(ppg_filtered, fs, 440, 1)  # Самый распростраённый случай
# res2 = p.process_data(ppg_filtered, fs, 44, 1)   # Окно маленького размера
# res3 = p.process_data(ppg_filtered, fs, 220, 22) # Большое окно с шагом в 10%
# res4 = p.process_data(ppg_filtered, fs, 4, 4)    # Мелкое окно с шагом в 100%

# for res in [res1, res2, res3, res4]:
#     print(res, '\n') # ПКМ --> Открыть в первичном обработчике данных


# Пример использования на чистых данных с пальца
fs = 120
ppg = []
fpath = __file__.split('/preprocess.py')[0] + '/examples/250409-Н-315-120.txt'
with open(fpath, 'r') as f:
    for line in f:
        ppg.append(float(line.strip()))

ppg_filtered = filtering.butter_bandpass(ppg, fs)
p = PreprocessPPG(vis=[
    'dists',
    'peaks',
    'lhf_plot',
    'lhf_comp',
    'outliers',
    'rsa',
    'seg',
    # 'seg_i'
])

res = p.process_data(ppg_filtered, fs, 240, 1, 'clear')
print(res) # ПКМ --> Открыть в первичном обработчике данных


# # Пример использования на шумных данных с запястья
# fs = 250
# # fs = 240
# fpath = __file__.split('/preprocess.py')[0] + '/examples/01_exp02_anxiety.csv'
# df = pd.read_csv(fpath)
# ppg = df["afe_LED1ABSVAL"].to_numpy()
# ppg = ppg[500:]

# p = PreprocessPPG(vis=[
#     # 'dists',
#     # 'peaks',
#     # 'hrv',
#     # 'lhf_plot',
#     # 'lhf_comp',
#     # 'rsa',
#     # 'outliers',
#     # 'seg',
#     # 'seg_i'
# ])

# res = p.process_data(ppg, fs, 375, 1, 'noisy')
# print(res) # ПКМ --> Открыть в первичном обработчике данных


__all__ = ["PreprocessPPG"]
