from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn import metrics
from abc import ABC, abstractmethod
import pandas as pd

from projet_ift712.modele import Modele

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


class svm(Modele):
    def __init__(self, data, features_names=0, features_nbr=0):
        # Appel du constructeur de la classe parente
        super().__init__(data, features_names, features_nbr, model=SVC(kernel='linear'))

    # Vous pouvez ajouter des méthodes spécifiques à la classe svm ici


# Créez une instance de la classe svm
svm_instance = svm(data, features_names, features_nbr)
X_train, X_test, y_train, y_test = svm_instance.split_data(data)
svm_instance.train(X_train, y_train)
svm_instance.evaluate_model(X_test, y_test)
