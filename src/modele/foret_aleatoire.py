from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit, learning_curve
import numpy as np

from modele.modele import Modele

class foret_aleatoire(Modele):
    def __init__(self, data, features_names, features_nbr, model):
        super().__init__(data=data, features_names=features_names, features_nbr=features_nbr)
        self.model = model
    
    def random_hyper_parameters_search(self, X_train, y_train, seed):
        """ Permet de rechercher les hyperparamètres optimaux à l'aide d'une recherche aléatoire avec de grands intervalles.
        Affiche les meilleurs hyperparamètres et retourne le modèle associé.

        :param X_train: Tableau (N, D) contenant les données d'entraînement
        :param y_train: Tableau (N,1) contenant les label des données d'entraînement
        
        :return: la forêt aléatoire avec les meilleurs hyperparamètres
        """
        
        # Les hyperparamètres
        param = {
            'n_estimators': [int(x) for x in np.linspace(10, 1000, 50)],
            'max_depth': [None] + [int(x) for x in np.linspace(10, 110, 10)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        # Recherche aléatoire
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        random_search = RandomizedSearchCV(self.model, param_distributions=param, cv=cv, scoring='accuracy', random_state=seed)
        random_search.fit(X_train, y_train)
        
        # Les meilleurs hyperparamètres
        best_estimator = random_search.best_estimator_
        print(f'La meilleure précision trouvée est de {random_search.best_score_} pour les hyperparamètres suivants : {random_search.best_params_}')
        return best_estimator

    def plot_learning_curves(self, X_train, y_train, train_size):
        N, train_score, val_score = learning_curve(self.model, X_train, y_train,train_sizes=train_size)
        plt.plot(N,train_score.mean(axis=1),label="Score d'entraînement")
        plt.plot(N,val_score.mean(axis=1),label="Score de validation")
        plt.xlabel("Taille de l'ensemble d'entraînement")
        plt.legend()
        

    
    def hyper_parameters_search(self, X_train, y_train, seed):
        """ Permet de rechercher les hyperparamètres optimaux à l'aide d'une recherche exhaustive, avec des invervalles pertinents, trouvés grâce à la recherche aléatoire.
        Affiche les meilleurs hyperparamètres et retourne le modèle associé.

        :param X_train: Tableau (N, D) contenant les données d'entraînement
        :param y_train: Tableau (N,1) contenant les label des données d'entraînement
        
        :return: la forêt aléatoire avec les meilleurs hyperparamètres
        """
        
        # Les hyperparamètres
        param = {
            'n_estimators': [int(x) for x in np.linspace(100, 200, 10)],
            'max_depth': [None] + [int(x) for x in np.linspace(50, 60, 1)],
            'min_samples_split': [9, 10, 11, 12],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True]
        }
        
        # Recherche stratifiée et exhaustive
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
        grid = GridSearchCV(self.model, param_grid=param, cv=cv, scoring='accuracy')
        grid.fit(X_train, y_train)

        # Le modèle avec les meilleurs paramètres
        best_estimator = grid.best_estimator_
        print(f'La meilleure précision trouvée est de {grid.best_score_} pour les hyperparamètres suivants : {grid.best_params_}')
        return best_estimator