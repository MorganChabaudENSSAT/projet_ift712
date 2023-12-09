from abc import ABC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics


class Modele(ABC):
    def __init__(self, data, features_names=0, features_nbr=0, model=None):
        self.model = model  # estimateur sklearn correspondant au modèle
        self.data = data    # dataframe contenant l'ensemble des données
        self.features_names = features_names    # Str_list conteannt les noms des featiures
        self.features_nbr = features_nbr    # entier : nombre de features

    def predict(self, X):
        """Predit la liste des classes de des données X passée en paramètres de la fonction.
        Cette fonction présuppose que la fonction train a été appelée auparavant

        :param X: Dataframe (N,D) contenant les échantillons
        :return: La liste des classes prédites par le modèle de regréssion logistique
        """
        # Prédiction
        y_pred = self.model.predict(X)
        return y_pred

    def train(self, X, y):
        """Entraîne le modèle sur les données passée en paramètre

        :param X: Dataframe contenant les données d'entraînement
        :param y: Dataframe de labels pour l'entraînement
        :return: None
        """
        self.model.fit(X, y)


    def scale_data(self, features_to_normalise, features_to_standardise):
        # TODO : Boucler sur les features à changer pour améliorer la modularité
        """Met à l'échelle les données.

           Cette fonction normalise les données qui ne suivent pas une distribution gaussienne et
           standardise les données qui en suivent une mais dont les valeurs sont trop petites ou trop grandes
           par rapport aux autres features risquant de fausser artificiellement l'évaluation du modèle.
           Elle utilise les Scaler de la bibliothèque sklearn

           Paramètres
           ------
           features_to_normalise :  liste de string contenant les features devant être normalisées
                                    (suite à la visualisation des données)
           features_to_standardise : liste de string contenant les features devant être normalisées
                                     (suite à la visualisation des données)
           Retourne
           -------
           scaled_data : dataframe contenant les données mise à l'échelle
           """
        mms = MinMaxScaler()  # Normalisation
        ss = StandardScaler()  # Standardisation

        scaled_data = self.data

        for feature_name in features_to_normalise :
            scaled_data[feature_name] = mms.fit_transform(scaled_data[[feature_name]])

        for feature_name in features_to_standardise :
            scaled_data[feature_name] = ss.fit_transform(scaled_data[[feature_name]])

        return scaled_data


    def split_data(self, data, test_size=0.20):
        """Sépare les données en 4 dataframes conteant les données d'entraînement et les labels associés ainsi que les
        données de tests avec les labels associés

        :param data:  dataframe contenant l'ensemble des données
        :param test_size : float indiquant le pourcentage de répartition des données entre les ensembles d'entraînement
        et de test. 80% pour l'entraîenemnt et 20% pour les tests par défaut

        :return:
        X_train : Dataframe de taille (test_size*N, D) contenant les données d'entrainement
        X_test : Dataframe de taille (test_size*N, D) contenant les données de test
        y_train : labels associés aux données d'entraîenemn,t
        y_test : labels associés aux données d'entraînement
        """

        # Tableux de features et de label
        X = data[self.features_names[:-1]]  # Features
        Y = data[self.features_names[-1]]  # Classes
        # Séparation des données en un ensemble de test (20%) et d'entraînement: (80%)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=16)
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, X, y, confusion_matrix=True, accuracy=True):
        """Evalue le modèle sommairement en affichant la matrice de confusion et l'accuracy associée à la prédiction du
        modèle sur l'ensmeble de données fourni

        :param X: Dataframe contenant les données à prédire
        :param y: Liste des labels des données
        :param confusion_matrix: Booléen indiquant si il faut afficher la matric de confusion
        :param accuracy: Booléen indiquant s'il faut afficher l'accuray
        :return:
        """
        y_pred = self.predict(X)
        if confusion_matrix:
            confusion_mtrx = metrics.confusion_matrix(y, y_pred)
            print(confusion_mtrx)
        if accuracy:
            accu = metrics.accuracy_score(y, y_pred)
            print(f"accuracy du modèle : {accu}")