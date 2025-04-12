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
        super().__init__(*args, **kwargs)

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

        self.mode = 'train'

    def forward(self, parameters: np.ndarray, diseases=None):
        if self.mode == 'train':
            for d in self.models.keys():
                disease_array = diseases.astype(np.object_)
                disease_array[diseases != d] = 0
                disease_array[diseases == d] = 1

                self.models[d].fit(parameters, disease_array.astype(np.int32))

        elif self.mode == 'test':
            patients = [{d: None for d in self.models.keys()} for _ in range(len(parameters))]
            for d in self.models.keys():
                preds = self.models[d].predict(parameters)
                for i in range(len(patients)):
                    patients[i][d] = preds[i]

            if type(diseases) == type(None):
                return patients
            else:
                score_report = []
                for d in self.models.keys():
                    disease_array = diseases.astype(np.object_)
                    disease_array[diseases != d] = 0
                    disease_array[diseases == d] = 1

                    score_report.append(f'{d} accuracy is {sklearn.metrics.accuracy_score(disease_array.astype(np.int32), [patients[_][d] for _ in range(len(patients))])}')

                return score_report

    def set_mode(self, mode):
        assert mode in ['train', 'test']

        self.mode = mode