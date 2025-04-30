
# 🧠 Deep Learning appliqué à la Cybersécurité pour la Détection d’Intrusion

## 📌 Contexte

Face à la complexité croissante des réseaux informatiques et à la multiplication des menaces cyber, les outils traditionnels de détection d’intrusion (comme Snort ou Suricata) atteignent leurs limites. Leur dépendance aux signatures rend difficile la détection d’attaques inconnues (zero-day). C’est dans ce contexte que les techniques de **Deep Learning** se positionnent comme une alternative puissante, capable de détecter des anomalies complexes et d’apprendre à partir de vastes ensembles de données.

## ❗ Problématique

Comment concevoir un modèle de détection d'intrusion capable de :
- Détecter en **temps quasi réel** des attaques réseau, notamment **DDoS**, même si elles ne sont pas référencées dans une base de signatures ?
- Exploiter efficacement un **volume massif de données réseau** avec des caractéristiques très variées ?
- **Améliorer la précision** des détections par rapport aux méthodes classiques tout en **réduisant les faux positifs** ?

## 🧪 Environnement de test (Testbed)

Le testbed repose sur une infrastructure simulée composée de deux réseaux :
- Un **Attack Network** générant différents types d’attaques (Brute Force SSH, DDoS, Botnet, etc.).
- Un **Victim Network** sécurisé, capturant le trafic réseau pendant **5 jours continus**.

Les paquets réseau sont traités avec **CICFlowMeter** pour extraire 80+ caractéristiques statistiques.  
Les fichiers `.pcap` sont convertis en `.csv` via cette solution Java, permettant l’analyse bidirectionnelle (BiFlow).

## 📂 Description du dataset

- **Source** : [CICIDS 2017 - Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)  
- **Volume** : 1 580 215 enregistrements  
- **Type d’attaques** : DDoS, Brute Force, Botnet, Web Attack, Infiltration, etc.  
- **Focus de l’étude** : Attaques **DDoS** détectées le **vendredi après-midi** (~225 745 exemples)  
- **Format** : 83 colonnes extraites des flux réseau TCP/UDP (durée, tailles de paquets, intervalles, flags...)

## ⚙️ Technologies utilisées

- **Langage** : Python  
- **Librairies** : `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`, `Keras`, `TensorFlow`  
- **IDE** : Jupyter Notebook, Spyder  
- **Environnement** : Anaconda, Google Colab  
- **Prétraitement** : CICFlowMeter, RFE pour la sélection de features

## 🧠 IA & Algorithmes utilisés

### 🔍 Prétraitement
- **Nettoyage et normalisation** des données
- **Analyse des données manquantes** (aucune observée)
- **Feature Selection** par **Recursive Feature Elimination (RFE)** pour extraire les attributs les plus pertinents :
  - `Flow-Duration`
  - `Flow IAT Std`
  - `Average Packet Size`
  - `Bwd Packet Length Std`

### 🏗️ Modèle IA : Deep Learning
- **Architecture** : Modèle séquentiel Keras avec plusieurs couches denses
- **Fonction de coût** : `binary_crossentropy`
- **Optimiseur** : `Adam`
- **Évaluation** : Matrice de confusion, précision, rappel, F1-score
- **Hyperparamètres** :
  - `Epochs = 120`
  - `Batch size = 1`
  - `Metrics = accuracy`

## 📈 Résultats

| Mesure               | Résultat     |
|----------------------|--------------|
| Précision            | **97 %**     |
| Détection DDoS       | Très fiable  |
| Faux positifs        | Faibles      |
| Avantage principal   | Détection anomalies complexes sans signatures |

## ✅ Contributions

- Construction d’un **modèle de Deep Learning** pour la détection d’attaques réseau
- **Utilisation d’un dataset réel et récent** avec divers scénarios d’intrusion
- Comparaison avec des outils classiques (Snort, Suricata)
- Reproductibilité assurée avec des scripts Python

## 🚀 Perspectives

- Intégration d’un système temps réel basé sur **Edge AI**
- Amélioration des performances avec **CNN + LSTM**
- Extension aux attaques de type malware, spyware et infiltration ciblée
- Déploiement dans une architecture **SIEM** (Splunk, ELK)


---

## 📝 Remarque importante

> **N.B** : Si votre système est insuffisant, je vous recommande humblement de ne pas aller plus loin, car le programme risque de ne pas fonctionner efficacement et vous risquez de perdre beaucoup de temps.

Ce projet repose sur **deux fichiers principaux** :
- `data_processing.py` : utilisé pour le **prétraitement des données**, les **représentations graphiques**, et la **sélection des attributs pertinents**.
- `construction_DeepLearning.py` : contient le code de **construction du modèle de Deep Learning**.

➡️ Une fois ces étapes préparatoires réalisées, vous pouvez **lancer l'entraînement du modèle**.
