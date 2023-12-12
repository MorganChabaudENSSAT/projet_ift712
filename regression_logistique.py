from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics
class RegressionLogistique(object):
    def __init__(self, data, features_names=0, features_nbr=0):
        self. logreg = LogisticRegression(random_state=16, solver='newton-cg')
        self.data = data
        self.features_names = features_names
        self.features_nbr = features_nbr

    def recherche_hyper_parametres(self, X, y, cv_iter=5):
        validation_scores = []
        solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
        for s in solvers :
            score = cross_val_score(LogisticRegression(random_state=16), X, y, cv=cv_iter).mean()
            validation_scores.append(score)
        plt.plot(solvers, validation_scores)
    def train(self, X_train, y_train, cross_val=False):
        self.logreg.fit(X_train, y_train)

    def predict(self, X):
        ''' Predit la classe de l'échantillon.
            Cette fonction présuppose que la fonction train a été appelée auparavant

            Paramètres
            ------
            X : vecteur contenant un échantillon

            Retourne
            ------
            La classe prédite par le modèle de regréssion logistique
        '''
        # Prédiction
        y_pred = self.logreg.predict(X)
        return y_pred

    def scale_data(self):
        # TODO : Boucler sur les features à changer pour améliorer la modularité
        """Met à l'échelle les données.

           Cette fonction normalise les données qui ne suivent pas une distribution gaussienne et
           standardise les données qui en suivent une mais dont les valeurs sont trop petites ou trop grandes
           par rapport aux autres features.
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
        ss = StandardScaler()  # Standardssation

        scaled_data = self.data
        #scaled_data['Oldpeak'] = mms.fit_transform(scaled_data[['Oldpeak']])

        scaled_data['Age'] = ss.fit_transform(scaled_data[['Age']])
        scaled_data['RestingBP'] = ss.fit_transform(scaled_data[['RestingBP']])
        scaled_data['Cholesterol'] = ss.fit_transform(scaled_data[['Cholesterol']])
        scaled_data['MaxHR'] = ss.fit_transform(scaled_data[['MaxHR']])

        return scaled_data
    def split_data(self, data):
        # Séparation des données en un ensemble d'entraînement et de test
        # Tableux de features et de label
        X = data[self.features_names[:-1]]  # Features
        Y = data[self.features_names[-1]]  # Classes
        # Séparation des données en un ensemble de test (20%) et d'entraînement: (80%)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=16)
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, X, y, cv_iter=5 ,confusion_matrix = True, accuracy = True):
        y_pred = self.predict(X)
        print(self.logreg.score(X, y))

        scores = cross_val_score(LogisticRegression(random_state=16), X, y, cv=cv_iter)
        print('Le score de validation croisée est : ', str(scores.mean()), 'avec un déviation standard de : ', str(scores.std()))

        if confusion_matrix:
            confusion_mtrx = metrics.confusion_matrix(y, y_pred)
            print(confusion_mtrx)
        if accuracy:
            accu = metrics.accuracy_score(y,y_pred)
            print(f"accuracy du modèle : {accu}")