from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, learning_curve
import numpy as np

from modele import Modele

class bayes_gaussien_naif(Modele):
    def __init__(self, data, features_names, features_nbr, model):
        super().__init__(data=data, features_names=features_names, features_nbr=features_nbr)
        self.model = model

    def plot_learning_curves(self, X_train, y_train, train_size):
        N, train_score, val_score = learning_curve(self.model, X_train, y_train,train_sizes=train_size)
        plt.plot(N,train_score.mean(axis=1),label="Score d'entraînement")
        plt.plot(N,val_score.mean(axis=1),label="Score de validation")
        plt.xlabel("Taille de l'ensemble d'entraînement")
        plt.legend()
        
    def hyper_parameters_search(self, X_train, y_train, seed):
        """
        Affiche les meilleurs hyperparamètres et retourne le modèle associé.

        :param X_train: Tableau (N, D) contenant les données d'entraînement
        :param y_train: Tableau (N,1) contenant les label des données d'entraînement
        
        :return: le Bayes gaussien naïf avec les meilleurs hyperparamètres
        """
        
        # Les hyperparamètres
        param = {
            'var_smoothing': np.logspace(0,-9, num=100)
        }
        
        # Recherche stratifiée et exhaustive
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        grid = GridSearchCV(self.model, param_grid=param, cv=cv, scoring='accuracy')
        grid.fit(X_train, y_train)

        # Le modèle avec les meilleurs paramètres
        best_estimator = grid.best_estimator_
        print(f'La meilleure précision trouvée est de {grid.best_score_} pour les hyperparamètres suivants : {grid.best_params_}')
        return best_estimator