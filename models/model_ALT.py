import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier


class MyCatBoostModel:
    def __init__(self, iter, diseases, *args, **kwargs):
        """Инициализация модели
        diseases - болезни, которые будет классифицировать модель"""

        self.a = {
            "iterations": iter,
            "depth": 8,
            "learning_rate": 0.03,
            "verbose": 100,
            "random_seed": 190525,
            "task_type": "GPU"
        }

        self.models = { d: CatBoostClassifier(**self.a) for d in diseases }
        self.mode = 'train'


    def probas_mode(self, probas, diseases):
        """Проверка необходимости вернуть вероятности и отсутствие в таком
        случае списка заболеваний"""

        if probas and type(diseases) != type(None):
            return False
        else:
            return True


    def forward(self, train_X: np.ndarray, train_y=None, val_X=None, val_y=None, probas=False):
        """Метод классификации и обучения модели
        train_X - массив параметров, необходимых для классификации
        diseases - параметр с дефолтным аргументом. Его необходимо задать
        для обучения, в режиме теста он необязателен, но если его передать,
        на выходе вы получите точность модели
        probas - параметр с дефолтным аргументом. Его необходимо задать, если
        вы собираетесь получить вероятности наличия заболеваний, однако его
        нельзя использовать вместе с diseases.        
        Метод возвращает классифицированное состояние пациента, вероятности
        наличия этих состояний или метрику точности"""

        assert self.probas_mode(probas, train_y)

        if self.mode == 'train':
            # Проходимся по каждой болячке
            for d in self.models.keys():
                # Размечаем данные по типу "Один против всех"
                train_array = train_y.astype(object)
                train_array[train_y != d] = 0
                train_array[train_y == d] = 1
                train_array = train_array.astype(np.int32)

                if val_X is not None and val_y is not None:
                    val_array = val_y.astype(object)
                    val_array[val_y != d] = 0
                    val_array[val_y == d] = 1
                    val_array = val_array.astype(np.int32)
                    eval_set = (val_X, val_array)
                else:
                    eval_set = None

                # Обучаем каждый метод в ансамбле, соответственно, классифицировать
                # целевые данные и отличать их от других
                print(f'Начинаю 1vA-тренировку для {d}...')
                self.models[d].fit(
                    train_X,
                    train_array,
                    eval_set=eval_set,
                    early_stopping_rounds=500
                )
                print(f'Тренировка весов для {d} завершена!\n')

        elif self.mode == 'test':
            # Каждый тестируемый пациент - словарь болячек, которыми они могут страдать
            patients = [{d: None for d in self.models.keys()} for _ in range(len(train_X))]

            # Проходимся по каждой болячке
            for d in self.models.keys():
                # Предсказываем наличие болячки на имеющихся параметрах
                if probas:
                    preds = self.models[d].predict_proba(train_X)[:, 1]
                else:
                    preds = self.models[d].predict(train_X)
                # Для каждой болячки в словаре пациента даём полученный вердикт
                for i in range(len(patients)):
                    patients[i][d] = preds[i]

            # Если мы оставили параметр diseases пустым, возвращаем результат работы модели
            if type(train_y) == type(None):
                return patients

            # Иначе также определяем массивы целевых меток по принципу "один против всех"
            # и высчитываем точность для каждой болячки
            else:
                score_report = []
                f1_report = []
                precision_report = []
                recall_report = []
                specificity_report = []

                for d in self.models.keys():
                    train_array = train_y.astype(np.object_)
                    train_array[train_y != d] = 0
                    train_array[train_y == d] = 1

                    cm = sklearn.metrics.confusion_matrix(
                        train_array.astype(np.int32),
                        [patients[_][d] for _ in range(len(patients))]
                    )

                    tn, fp, fn, tp = cm.ravel()
                    precision = tp/(tp+fp)
                    recall = tp/(tp+fn)
                    specificity = tn/(tn+fp)

                    score_report.append(f'{d} accuracy is {(tp+tn)/(tp+tn+fp+fn)}')
                    f1_report.append(f'{d} f1-score is {2*(precision*recall)/(precision+recall)}')
                    precision_report.append(f'{d} precision is {precision}')
                    recall_report.append(f'{d} recall is {recall}')
                    specificity_report.append(f'{d} specificity is {specificity}')

                return score_report, f1_report, precision_report, recall_report, specificity_report
            

    def get_importances(self):
        return [self.models[d].feature_importances_ for d in self.models.keys()]


    def save_weights(self, path):
        for d in self.models:
            self.models[d].save_model(f'{str(path) + "__" + str(d) + ".cbm"}')
            print(f'Веса для {d} сохранены в "{str(path) + "__" + str(d) + ".cbm"}"')


    def load_weights(self, path):
        for d in self.models.keys():
            model = CatBoostClassifier(**self.a)
            model.load_model(f'{str(path) + "__" + str(d) + ".cbm"}')
            self.models[d] = model
            print(f'Веса для {d} загружены из "{str(path) + "__" + str(d) + ".cbm"}"')


    def set_mode(self, mode):
        """Устанавливает режим работы модели
        mode -- соответственно, режим. Может быть только train или test
        Во всех иных случаях метод выдаёт ошибку"""

        assert mode in ['train', 'test']
        self.mode = mode
