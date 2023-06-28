import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import itertools
from sklearn.metrics import accuracy_score

class ParamSearch:
    def __init__(self, model, param_grid:dict, n_folds:int = 5) -> None:
        self.model = model

        # Set of the paremeters grid
        keys, values = zip(*param_grid.items())
        self.param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

        self.n_folds = n_folds
        self.CV = StratifiedKFold(n_splits = n_folds)

        self.cv_results_ = {
            "params": self.param_grid,
            "train_mean": np.zeros(1),
            "train_std": np.zeros(1),
            "test_mean": np.zeros(1),
            "test_std": np.zeros(1)
        }
        

    def fit(self, X:np.ndarray, y:np.ndarray):
        
        
        mean_acc_test = np.zeros(len(self.param_grid))
        std_acc_test = np.zeros(len(self.param_grid))
        mean_acc_train = np.zeros(len(self.param_grid))
        std_acc_train = np.zeros(len(self.param_grid))

        for i, d in enumerate(self.param_grid):

            temp_test_acc = np.zeros(self.n_folds)
            temp_train_acc = np.zeros(self.n_folds)

            for j, (train_index, test_index) in enumerate(self.CV.split(X, y)):
                X_train = X[train_index, :]
                y_train = y[train_index]

                X_test = X[test_index, :]
                y_test = y[test_index]

                # Change to a  Function
                # From here ---------------------------------------------------------
                self.model.set_params(**d)
                self.model.fit(X_train, y_train)

                y_pred_test = self.model.predict(X_test)
                y_pred_train = self.model.predict(X_train)

                temp_test_acc[j] = accuracy_score(y_test, y_pred_test)
                temp_train_acc[j] = accuracy_score(y_train, y_pred_train)
                # to here -----------------------------------------------------------
        
            mean_acc_test[i] = temp_test_acc.mean()
            std_acc_test[i] = temp_test_acc.std()
            mean_acc_train[i] = temp_train_acc.mean()
            std_acc_train[i] = temp_train_acc.std()

        self.cv_results_["train_mean"]=mean_acc_train
        self.cv_results_["train_std"]=std_acc_train
        self.cv_results_["test_mean"]=mean_acc_test
        self.cv_results_["test_std"]=std_acc_test

        return self
    

    def to_csv(self, name_file:str, path:str = ''):
        
        params_values = np.array([list(d.values())[0] for d in self.cv_results_["params"]])
        
        temp = np.array([
                params_values,
                self.cv_results_["train_mean"],
                self.cv_results_["train_std"],
                self.cv_results_["test_mean"],
                self.cv_results_["test_std"]]).T
        
        res = pd.DataFrame(temp, columns=["params", "train_mean", "train_std", "test_mean", "test_std"])

        res.to_csv(path+name_file, index=False)
        return self

if __name__ == "__main__":
    from sklearn.naive_bayes import GaussianNB
    from sklearn import datasets
    
    proc = GaussianNB()
    iris = datasets.load_iris()
    
    parameters = {'var_smoothing': (1e-9, 1)}

    search = ParamSearch(model=proc, param_grid=parameters, n_folds=5)
    search.fit(iris.data, iris.target)
    search.to_csv(r'C:/Users/dra98/OneDrive/Documentos/Trabajo/Doctorado/Codigo/src/tests/data.csv')
    print(search.cv_results_)
