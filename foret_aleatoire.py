from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from modele import Modele

#######################
# A RETIRER PLUS TARD #
#######################
# Importation des données
df = pd.read_csv('heart.csv')  # Dataframe contenant les données
features_names = df.columns
features_nbr = features_names.shape[0]
print(f"nombre de features dans le dataset : {features_nbr}")
# Visualisation des données pour mieux les comprendre
print(df.head())
print(df.dtypes)
le = LabelEncoder()

data = df.copy(deep=True)

data['Sex'] = le.fit_transform(data['Sex'])
data['ChestPainType'] = le.fit_transform(data['ChestPainType'])
data['RestingECG'] = le.fit_transform(data['RestingECG'])
data['ExerciseAngina'] = le.fit_transform(data['ExerciseAngina'])
data['ST_Slope'] = le.fit_transform(data['ST_Slope'])

print(data.head())

class foret_aleatoire(Modele):
    def __init__(self, data, features_names=0, features_nbr=0, model=None):
        super().__init__(data=data, features_names=features_names, features_nbr=features_nbr)
        self.model = model
        
    def hyper_parameters_search(self, X_train, y_train):
        """
        Affiche les meilleurs hyperparamètres et retourne le modèle associé.

        :param X_train: Tableau (N, D) contenant les données d'entraînement
        :param y_train: Tableau (N,1) contenant les label des données d'entraînement
        
        :return: la forêt aléatoire avec les meilleures hyperparamètres dépendemment du noyau utilisé
        """
        
        # Définissez l'espace des hyperparamètres à explorer
        param = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Instanciez le modèle DecisionTreeClassifier
        rf_model = RandomForestClassifier(criterion='log_loss')

        # Recherche aléatoire
        random_search = RandomizedSearchCV(rf_model, param_distributions=param, n_iter=100, cv=5, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        
        # Recherche stratifiée et exhaustive
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(self.model, param_grid=param, cv=cv, scoring='accuracy')
        grid.fit(X_train, y_train)

        # Obtenez les meilleurs hyperparamètres
        best_params = random_search.best_params_
        print(f"Best Hyperparameters: {best_params}")
        print(f'et {random_search.best_score_}')

        # Utilisez le modèle avec les meilleurs paramètres
        best_estimator = grid.best_estimator_
        print(f'La meilleure accuracy trouvée est de {grid.best_score_} pour les hyperparamètres suivants : {grid.best_params_}')
        return best_estimator