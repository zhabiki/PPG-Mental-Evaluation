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

class SuicideInclinationClassifier:
    def __init__(self, diseases, method, *args, **kwargs):
        """Инициализация модели
        diseases - болезни, которые будет классифицировать модель
        method - метод, который мы используем для обучения и классификации"""
        super().__init__(*args, **kwargs)

        #Выбираем метод
        if method == 'xgboost':
            self.models = {d: xgboost.XGBClassifier() for d in diseases}
        elif method == 'gradboost':
            self.models = {d: sklearn.ensemble.GradientBoostingClassifier() for d in diseases}
        elif method == 'adaboost':
            self.models = {d: sklearn.ensemble.AdaBoostClassifier() for d in diseases}
        elif method == 'LDA':
            self.models = {d: sklearn.discriminant_analysis.LinearDiscriminantAnalysis() for d in diseases}
        elif method == 'SVM':
            self.models = {d: sklearn.svm.LinearSVC() for d in diseases}
        elif method == 'QDA':
            self.models = {d: sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis() for d in diseases}
        elif method == 'LR':
            self.models = {d: sklearn.linear_model.LogisticRegression() for d in diseases}
        elif method == 'DT':
            self.models = {d: sklearn.tree.DecisionTreeClassifier() for d in diseases}
        elif method == 'NB':
            self.models = {d: sklearn.naive_bayes.GaussianNB() for d in diseases}

        #Устанавливаем режим тренировки, ибо модель перед исполльзованием надо натренировать
        self.mode = 'train'

    def forward(self, parameters: np.ndarray, diseases=None):
        """Метод классификации и обучения модели
        parameters - массив параметров, необходимых для классификации
        diseases - параметр с дефолтным аргументом. Его необходимо задать
        для обучения, в режиме теста он необязателен, но если его передать,
        на выходе вы получите точность модели
        
        Метод возвращает классифицированное состояние пациента или метрику точности"""
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
                for d in self.models.keys():
                    disease_array = diseases.astype(np.object_)
                    disease_array[diseases != d] = 0
                    disease_array[diseases == d] = 1

                    score_report.append(f'{d} accuracy is {sklearn.metrics.accuracy_score(disease_array.astype(np.int32), [patients[_][d] for _ in range(len(patients))])}')

                return score_report

    def set_mode(self, mode):
        """Устанавливает режим работы модели
        mode - соответственно, режим. Может быть только train или test
        Во всех иных случаях метод выдаёт ошибку"""
        assert mode in ['train', 'test']

        self.mode = mode