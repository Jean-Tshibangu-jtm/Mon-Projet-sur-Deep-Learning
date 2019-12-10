# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:04:55 2019

@author: X240
"""
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Activation


# Nous utilisons la bibliothèque du pandas pour charger des données et examiner la forme de notre ensemble de données


da = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
#data_set_DDoS = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
data_set_DDoS = da.iloc[77200:77900,]
#data_set_DDoS = da.iloc[77000:77600,]

#Fractionnement des attributs pertinents et tags

#Variable explicative
#attributs_pertinents_DDoS= data_set_DDoS.iloc[:,7:84]
attributs_pertinents_DDoS= data_set_DDoS.iloc[:,[7,19,23,58]]
#Variable cible
attributs_tag= data_set_DDoS.iloc[:,-1]

#labelisation des variables catégoriques
#LabelEncoder	Encode les étiquettes avec une valeur comprise entre 0 et n_classes-1
labEncr_attributs_tag = LabelEncoder()

attributs_tag = labEncr_attributs_tag.fit_transform(attributs_tag)

#  scinder le jeu de données maintenant
#Franctionnement de données
#XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)
attributs_pertinents_DDoS_train,attributs_pertinents_DDoS_test,attributs_tag_train,attributs_tag_test = train_test_split(attributs_pertinents_DDoS,attributs_tag, test_size = 0.2, random_state = 0)

#attributs_explicatifs_train,attributs_explicatifs_test,attribut_tag_train,attribut_tag_test=train_test_split(attributs_explicatifs,attribut_tag,test_size=0.3,random_state=0,stratify=attribut_tag)


# mise à l'échelle des fonctionnalités
#standardisation (centrer-réduire )" signifie conversion vers un standard commun

StdSc = StandardScaler()
attributs_pertinents_DDoS_train = StdSc.fit_transform(attributs_pertinents_DDoS_train)



attributs_pertinents_DDoS_test = StdSc.fit_transform(attributs_pertinents_DDoS_test)

#construire le modèle
#Méthodes pour optimiseur
#initialisation du modele
#( nombre d'entités + nombre de nœuds de sortie / 2 )
#model = Sequential([ Dense(32, input_shape=(784,)),Activation('relu'),Dense(10),Activation('softmax'),



def classifier(optimizer):
    model = Sequential()
    model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=4))
    model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

    # choisir des hyperparamètres 
model = KerasClassifier(build_fn=classifier)    
    
params = {'batch_size': [1, 5],'epochs': [100, 120], 'optimizer': ['adam', 'rmsprop']}

gridSearch = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=10)

gridSearch = gridSearch.fit(attributs_pertinents_DDoS_train, attributs_tag_train)




score = gridSearch.best_score_
bestParams = gridSearch.best_params_


print('best_accuracy_score:',score)
print('best_parameters:',  bestParams)



#model = Sequential()
#model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=29))
#model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#model = Sequential()
#model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=29))
#model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#Construction du modèle
    model = Sequential()
    model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=4))
    model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#  attributs_pertinents_DDoS_train=attributs_pertinents_DDoS_train.astype(float)
#  attributs_tag_train=attributs_tag_train.astype(float)

#Compilez le classifieur
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Maintenant ajustons sur les données.
#model.fit(XTrain, yTrain, batch_size=1, epochs=120)
model.fit(attributs_pertinents_DDoS_train, attributs_tag_train, batch_size=1, epochs=120)



#Evaluation du modèle

#yPred = model.predict(XTest)
tag_Pred = model.predict(attributs_pertinents_DDoS_test)

#yPred = [1 if y > 0.5 else 0 for y in yPred]
tag_Pred = [1 if attributs_tag > 0.5 else 0 for attributs_tag in tag_Pred]

#matrix = confusion_matrix(yTest, yPred)

matrice_de_confusion = confusion_matrix(attributs_tag_test, tag_Pred)

#accuracy = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
#print("Accuracy: " + str(accuracy * 100) + "%")


accuracy = (matrice_de_confusion[0][0] + matrice_de_confusion[1][1]) / (matrice_de_confusion[0][0] + matrice_de_confusion[0][1] + matrice_de_confusion[1][0] + matrice_de_confusion[1][1])
print("Accuracy: " + str(accuracy * 100) + "%")




