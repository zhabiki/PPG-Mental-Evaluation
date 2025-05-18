import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.svm
import sklearn.metrics
import sklearn.tree
import xgboost
import sklearn
import numpy as np
import matplotlib.pyplot as plt

class PsychiatryDiseasesClassifier:
    def __init__(self, diseases: list, *args, **kwargs):
        """Инициализация модели
        diseases - болезни, которые будет классифицировать модель"""

        self.models = {d: sklearn.ensemble.GradientBoostingClassifier() for d in diseases}

        #Устанавливаем режим тренировки, ибо модель перед исполльзованием надо натренировать
        self.mode = 'train'

    def probas_mode(self, probas: bool, diseases: bool):
        """Проверка необходимости вернуть вероятности и отсутствие в таком
        случае списка заболеваний"""
        if probas and type(diseases) != type(None):
            return False
        else:
            return True

    def forward(self, parameters: np.ndarray, diseases=None, probas=False):
        """Метод классификации и обучения модели
        parameters - массив параметров, необходимых для классификации
        diseases - параметр с дефолтным аргументом. Его необходимо задать
        для обучения, в режиме теста он необязателен, но если его передать,
        на выходе вы получите точность модели
        probas - параметр с дефолтным аргументом. Его необходимо задать, если
        вы собираетесь получить вероятности наличия заболеваний, однако его
        нельзя использовать вместе с diseases
        
        Метод возвращает классифицированное состояние пациента, вероятности
        наличия этих состояний или метрику точности"""
        assert self.probas_mode(probas, diseases)

        #Если тренировка
        if self.mode == 'train':
            #Проходимся по каждой болячке
            for d in self.models.keys():
                #Размечаем данные по типу "Один против всех"
                disease_array = diseases.astype(np.object_)
                disease_array[diseases != d] = 0
                disease_array[diseases == d] = 1

                #Обучаем каждый метод в ансамбле, соответственно, классифицировать целевые данные и отличать их от других
                self.models[d].fit(parameters, disease_array.astype(np.int32))
        #Если мы тестируем модель
        elif self.mode == 'test':
            #Каждый тестируемый пациент - словарь болячек, которыми они могут страдать
            patients = [{d: None for d in self.models.keys()} for _ in range(len(parameters))]
            #Проходимся по каждой болячке
            for d in self.models.keys():
                #Предсказываем наличие болячки на имеющихся параметрах
                if probas:
                    preds = self.models[d].predict_proba(parameters)
                else:
                    preds = self.models[d].predict(parameters)
                #Для каждой болячки в словаре пациента даём полученный вердикт
                for i in range(len(patients)):
                    patients[i][d] = preds[i]

            #Если мы оставили параметр diseases пустым, возвращаем результат работы модели
            if type(diseases) == type(None):
                return patients
            #Иначе также определяем массивы целевых меток по принципу "один против всех" и высчитываем точность для каждой болячки
            else:
                score_report = []
                f1_report = []
                precision_report = []
                recall_report = []
                specificity_report = []
                for d in self.models.keys():
                    disease_array = diseases.astype(np.object_)
                    disease_array[diseases != d] = 0
                    disease_array[diseases == d] = 1

                    cm = sklearn.metrics.confusion_matrix(disease_array.astype(np.int32), [patients[_][d] for _ in range(len(patients))])
                    tn, fp, fn, tp = cm.ravel()

                    precision = tp/(tp+fp)
                    recall = tp/(tp+fn)
                    specificity = tn/(tn+fp)

                    score_report.append(f'{d} accuracy is {(tp+tn)/(tp+tn+fp+fn)}')
                    f1_report.append(f'{d} f1-score is {2*(precision*recall)/(precision+recall)}')
                    precision_report.append(f'{d} precision is {precision}')
                    recall_report.append(f'{d} recall is {recall}')
                    specificity_report.append(f'{d} specificity is {specificity}')

                    # score_report.append(f'{d} accuracy is {sklearn.metrics.accuracy_score(disease_array.astype(np.int32), [patients[_][d] for _ in range(len(patients))])}')
                    # f1_report.append(f'{d} accuracy is {sklearn.metrics.f1_score(disease_array.astype(np.int32), [patients[_][d] for _ in range(len(patients))])}')
                    # precision_report.append(f'{d} precision is {sklearn.metrics.precision_score(disease_array.astype(np.int32), [patients[_][d] for _ in range(len(patients))])}')
                    # recall_report.append(f'{d} accuracy is {sklearn.metrics.recall_score(disease_array.astype(np.int32), [patients[_][d] for _ in range(len(patients))])}')

                return score_report, f1_report, precision_report, recall_report, specificity_report
            
    def get_weights(self, features, vis=False):
        for d in self.models.keys():
            print(f"Для болезни {d} веса признаков равны " + str(self.models[d].feature_importances_) + '\n')
            if vis:
                plt.bar(features, self.models[d].feature_importances_)
                plt.title(f'Для болезни {d} веса признаков равны')
                plt.savefig(f'method_results/{d}_weights.png')
                plt.close()

    def set_mode(self, mode: str):
        """Устанавливает режим работы модели
        mode - соответственно, режим. Может быть только train или test
        Во всех иных случаях метод выдаёт ошибку"""
        assert mode in ['train', 'test']

        self.mode = mode
