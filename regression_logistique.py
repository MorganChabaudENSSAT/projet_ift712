class Regression_logistique(object):
    def __init__(self,X, Y):
        self.X = X
        self.Y = Y


# TODO : Implémenter ca bien
# Tableux de features et de label
X = data[columns_names[:-1]] # Features
Y = data[columns_names[-1]] # Classes

print(data) #Visulalisation


# Preparation des données
#Séparation des données en un ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=16)
# instanciation du modèle
logreg = LogisticRegression(random_state=16)

# entrainement
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# import the metrics class
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


