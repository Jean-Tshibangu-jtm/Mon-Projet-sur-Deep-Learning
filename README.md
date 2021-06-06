# Système de détection d'intrusion utilisant le Deep Learning

# Abstract

Avec la croissance exponentielle de la taille des réseaux informatiques et des applications développées, l'augmentation significative des dommages potentiels pouvant être causée par le lancement d'attaques devient évidente. Concurremment, les systèmes de détection d'intrusion (IDS) et les systèmes de prévention d'intrusion (IPS) sont des outils de détection et défenses les plus importantes contre les attaques réseau sophistiquées, et en croissance constante. 

Par conséquent, l'objectif de ce travail est d'appliquer une méthode de modélisation de données de pointe d’un système de détection d’intrusion pour prédire une détection d’intrusion dans un environnement réseau à l'aide de l’algorithme de l’apprentissage profond  comme outil de prédiction de l’attaque.
Pour ce faire, nous avons utilisé le jeu de données ISCX 2017 collecté par l'Institut canadien de cybersécurité. Cet ensemble de données contient sept flux de réseau d’attaques bénignes et communes, qui répond aux critères du monde réel, et est publiquement disponible. 
L'ensemble de données d'origine comprend 1580215 observations collectées en cinq jours avec une diversité d’attaques, dont 225745 observations le dernier pour l’attaque DDoS, et 85 fonctionnalités. Nous échantillonnons les données au hasard 10 fois avec la validation croisée pour obtenir 10 sous-échantillons de données permettant de créer des modèles de prédiction utilisant l’algorithme d’apprentissage profond, avec une architecture séquentielle.
Les modèles obtenus sont comparés à d'autres techniques proposés, pour évaluer les précisions sur la base de la matrice de confusion. La précision trouvée dans notre modèle est 97\%.
En conséquence, ce document évalue les performances d'un ensemble complet de fonctionnalités de trafic réseau avec l'algorithme d'apprentissage profond, afin de détecter l'attaque dans un réseau informatique.

# Technologies et Mise en oeuvre
Modèle d'apprentissage en profondeur VGG-19 formé à l'aide de l'ensemble de données de l’Institut Canadien pour la Cybersécurité ISCx 2017
# Framework et API
•	Tensorflow-GPU
•	Keras
# Outils
•	Anaconda (Python 3.6)
•	Spyder, Jupyter 
# Comment utiliser
Téléchargez l'ensemble de données ISCX 2017 à partir du lien
https://www.unb.ca/cic/datasets/ids-2017.html
# N.B : Si votre système est inadéquat, je vous demande humblement de vous arrêter ici car le programme ne fonctionnera pas efficacement et beaucoup de temps sera perdu.
Nous avons deux fichiers : data_processing.py et construction_DeepLearning.py. Le premier fichier est utilisé pour le prétraitement de données, représentations graphiques, vérifications des attributs pertinents, et le second fichier comprend les codes pour la construction du modèle.
Et vous pouvez commencer la formation.
# BONNE CHANCE ! 
