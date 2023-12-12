from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, learning_curve
from sklearn import preprocessing

from modele import Modele


class Svm(Modele):
    def __init__(self, data, features_names=0, features_nbr=0, model=None):
        Modele.__init__(
            self=self,
            data=data,
            features_names=features_names,
            features_nbr=features_nbr
        )
        match model:
            case 'linear':
                self.model = SVC(kernel='linear')
            case 'rbf':
                self.model = SVC(kernel='rbf')
            case 'sigmoid':
                self.model = SVC(kernel ='sigmoid')
            case 'poly' :
                self.model = SVC(kernel='poly')
            case _ :
                self.model = model
    
    def hyper_parameters_search(self, X_train, y_train):
        """Recherche les meilleurs hyperparamètres pour la SVM dépendemment du type de noyau utilisé par 'grid search'.
        Affiche les meilleurs hyperparamètres et retourne le modèle associé.

        :param X_train: Tableau (N, D) contenant les données d'entraînement
        :param y_train: Tableau (N,1) contenant les label des données d'entraînement

        :return: la svm avec les meilleures hyperparamètres dépendemment du noyau utilisé
        """
        if self.model.kernel == 'rbf':
            C_range = [1,10,100,1000]
            gamma_range = [1,0.1,0.001,0.0001]
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(self.model, param_grid=param_grid, cv=cv, scoring='accuracy')
            grid.fit(X_train, y_train)
            print(grid.best_score_, " pour les hyperparamètres suivants, noyau RBF : ", grid.best_params_)
            return grid.best_estimator_
        elif self.model.kernel == 'sigmoid':
            coef0_range = [-5, -3, -1, -0.1, 0.0, 0.1, 1]
            param_grid = dict(coef0=coef0_range)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(self.model, param_grid=param_grid, cv=cv, scoring='accuracy')
            grid.fit(X_train, y_train)
            print(grid.best_score_, " pour les hyperparamètres suivants, noyau sigmoïde : ", grid.best_params_)
            return grid.best_estimator_
        elif self.model.kernel == 'poly':
            coef0_range = [-5, -3, -1, -0.1, 0.0, 0.1, 1]
            degree_range = [2, 3, 4, 5, 6, 7]
            param_grid = dict(coef0=coef0_range, degree = degree_range)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(self.model, param_grid=param_grid, cv=cv, scoring='accuracy')
            grid.fit(X_train, y_train)
            print(grid.best_score_, " pour les hyperparamètres suivants, noyau polynomial : ", grid.best_params_)
            return grid.best_estimator_
        print("Erreur recherche hyper-paramètres")
        return self.model
    
    def plot_learning_curves(self, X_train, y_train, train_size):
        N, train_score, val_score = learning_curve(self.model, X_train, y_train,train_sizes=train_size)
        plt.plot(N,train_score.mean(axis=1),label="Score d'entraînement")
        plt.plot(N,val_score.mean(axis=1),label="Score de validation")
        plt.xlabel("Taille de l'ensemble d'entraînement")
        plt.legend()
        
        
        
