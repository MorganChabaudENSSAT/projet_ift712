import numpy as np
from matplotlib import pyplot as plt

from modele import Modele
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, learning_curve


class K_PP_voisins(Modele):
    def __init__(self, data, features_names, features_nbr, model=None):
        Modele.__init__(
            self=self,
            data=data,
            features_names=features_names,
            features_nbr=features_nbr
        )
        if model is None :
            self.model = KNeighborsClassifier(n_neighbors=5)
        else :
            self.model = model

    def K_fold(self,X_train, y_train,n_splits=10, n_repeats=3, scoring='accuracy'):
        """ Applique une cross-validation stratifiée

        :param X_train: Tableau contenant les données d'entraînement
        :param y_train: Tableau contenant les labels des données d'entraînement
        :param n_splits: Entier correspondant au nombre de splits dans la cross validation (10 par défaut)
        :param n_repeats: Entier : au nombre de fois que la cross-validation doit être répétée (3 par défaut)
        :param scoring: metrique utilisée pour al cross-validation
        :return: Affiche le score de cross validation.
        """
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        cv_score = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring).mean()
        print("Le score de cross-validation du modèle est: ",cv_score)

    def hyper_parameters_search(self, X_train, y_train, h_parameters_to_tune, other_parameters, cv=5):
        """Recherche les meilleurs hyperparamètres pour la régression logistique par 'grid search' basée sur une
        cross-validation. Affiche les meilleurs hyperparamètres et retourne le modèle associé.

        :param X_train: Tableau (N, D) contenant les données d'entraînement
        :param y_train: Tableau (N,1) contenant les label des données d'entraîenment
        :param h_parameters_to_tune: Dictionnaire contenant les hyperparamètres à tester ainsi que les valeurs qui
                                     leurs sont associées
        :param cv_iter: Nombre de répétitions dans le K-Fold

        :return: la régression avec les meilleures hyperparamètres
        """
        grid = GridSearchCV(KNeighborsClassifier(leaf_size=1), h_parameters_to_tune, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        print(grid.best_score_, " pour les hyperparamètres suivants : ", grid.best_params_)
        return grid.best_estimator_

    def data_needed_for_max_score(self, X_train, y_train, train_size):
        """
        :param X_train: Tableau (N, D) contenant les données d'entraînement
        :param y_train: Tableau (N,1) contenant les label des données d'entraîenment
        :param train_size: Entier correspondant au pourcentage de répartition des données entre les ensembles
        d'entraînement et de test
        :return: Affiche le graphique des scores des modèles en fonction du nombre de données d'entraînement
        """
        N, train_score, val_score = learning_curve(self.model, X_train, y_train,train_sizes=train_size, scoring="accuracy")
        plt.plot(N,train_score.mean(axis=1),label="score d'entraîenemnt")
        plt.plot(N,val_score.mean(axis=1),label="score de validation")
        plt.xlabel("taille de l'ensemble d'entraînement")
        plt.legend()