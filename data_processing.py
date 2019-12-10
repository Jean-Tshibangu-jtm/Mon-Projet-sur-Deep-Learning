# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:04:55 2019

@author: TSHIBA-PC
"""
import statsmodels as stat
#import seaborn as sbrn
import pandas as pds
import matplotlib.pyplot as mplt
import numpy as np 
#import sklearn.linear_model.LogisticRegression as LogisticRegression
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


#Import LogisticRegression for performing chi square test from sklearn.linear_model import LogisticRegression


#dtst = pds.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
data_set = pds.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
#
#data_set = pds.read_excel('dataSet-DDos.pcap_ISCX BIS.xls')
#data_set = pds.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
#data_set = pds.read_excel('dataSet-DDos.pcap_ISCX BIS.xls')
#data_set = pds.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
print(data_set.shape)

data_set.values

print("Correlation:", data_set.corr(method='pearson'))

Y= data_set.iloc[:,84].values

#Extrait ttes les observations(lignes) et 7,19,23,58 colonnes
X= data_set.iloc[:,7:84].values
#X= data_set.iloc[:,25:84].values


#test = SelectKBest(score_func=f_classif, k=4)

#fit = test.fit(X,Y)

#features = fit.transform(X)

#print(features[0:6,:])

#fit
#RFE, SELECTION DES ATTRIBUTS PERTINENTS
model = LogisticRegression()
rfe = RFE(model, 4)

fit = rfe.fit(X,Y)
fit

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s"% fit.support_) 
print("Feature Ranking: %s"% fit.ranking_)

#FIN RFE, SELECTION DES ATTRIBUTS PERTINENTS
#affiche les nombres d'observations et attributs du dataset
print(data_set.shape)
#affiche les infos telles que mean, ecart-type(std), max etc... et permet aussi de voir les valeurs manquantes
print(data_set.describe())
#affiche les noms des attributs
print(data_set.columns)
#Extrait ttes les observations que pour la dernière colonne.
attributs_tag= data_set.iloc[:,-1].values
#Extrait ttes les observations(lignes) et 7,19,23,58 colonnes
attributs_pertinents_DDoS= data_set.iloc[:,[7,19,23,58]].values
#Verifie les données nulles si ça existe c True, sinon false
print(data_set.isnull().sum)
#LabelEncoder	Encode les étiquettes avec une valeur comprise entre 0 et n_classes-1
labEncr_attributs_tag = LabelEncoder()
attributs_tag = labEncr_attributs_tag.fit_transform(attributs_tag)
#Franctionnement de données
attributs_pertinents_DDoS_train,attributs_pertinents_DDoS_test,attributs_tag_train,attributs_tag_test = train_test_split(attributs_pertinents_DDoS,attributs_tag,test_size = 0.2,random_state = 0)

#standardisation (centrer-réduire )" signifie conversion vers un standard commun

StdSc = StandardScaler()

attributs_pertinents_DDoS_train = StdSc.fit_transform(attributs_pertinents_DDoS_train)

attributs_pertinents_DDoS_test = StdSc.fit_transform(attributs_pertinents_DDoS_test)

#Normalisation (centrer-réduire )" signifie conversion vers un standard commun
attributs_pertinents_DDoS_train = normalize(attributs_pertinents_DDoS_train)
attributs_pertinents_DDoS_test = normalize(attributs_pertinents_DDoS_test)

#VISUALISATION EN HISTOGRAMME
#mplt.hist(attributs_pertinents_DDoS[:,0],bins=2)
mplt.hist(attributs_tag[:,],bins=2,rwidth=0.70, color='pink')
#Labeliser l'Axes et Titre
mplt.title('Echantillon Identifié')
mplt.ylabel('Echantillon')
mplt.xlabel('Distribution Attaque') 

donnee_ligne ={"attaque":["Normale","Anormale"]}
df= pds.DataFrame(donnee_ligne, columns=pds.Index(['attaque'],name='attribute'))
#mplt.show()
#mplt.colorbar()

# Style de l'arriere plan
mplt.style.use(['dark_background','fast'])

#legend
#mplt.legend(['Normale','Anormale'], loc=5)

# Formater la lgne de couleur
#mplt.plot(attributs_tag,'green')

# sauvegarde en format pdf 
mplt.savefig('Proportion Normale et Anormale.pdf', format='pdf')












