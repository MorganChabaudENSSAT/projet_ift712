from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from modele.modele import Modele


class Reseau_neurones(Modele):
    def __init__(self, data, features_names=0, features_nbr=0, model=None):
        Modele.__init__(
            self=self,
            data=data,
            features_names=features_names,
            features_nbr=features_nbr
        )
        if model is None:
            self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1,
                                       max_iter=500)
        else:
            self.model = model

    def hyper_parameters_search(self, X_train, y_train):
        """Recherche les meilleurs hyperparamètres pour le réseau de neurones.
        Affiche les meilleurs hyperparamètres et retourne le modèle associé.

        :param X_train: Tableau (N, D) contenant les données d'entraînement
        :param y_train: Tableau (N,1) contenant les labels des données d'entraînement

        :return: le réseau de neurones avec les meilleures hyperparamètres
        """
        param_grid = {
            #'solver': ['lbfgs','sgd','adam'],
            'solver': ['adam'],
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            # 'activation': ['relu', 'logistic', 'tanh'],
            'activation': ['tanh'],
            'alpha': [1e-3, 1e-4, 1e-5],
            'max_iter': [500]
        }
        grid = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        print(grid.best_score_, " pour les hyperparamètres suivants : ", grid.best_params_)
        return grid.best_estimator_
    
    def plot_learning_curves(self, X_train, y_train, train_size):
        N, train_score, val_score = learning_curve(self.model, X_train, y_train,train_sizes=train_size)
        plt.plot(N,train_score.mean(axis=1),label="score d'entraîenemnt")
        plt.plot(N,val_score.mean(axis=1),label="score de validation")
        plt.xlabel("taille de l'ensemble d'entraînement")
        plt.legend()

