# RabebKaabi_Servier
La prédiction des propriétés d'une molécule de médicament joue un rôle important dans le processus de conception de médicaments. Les propriétés de la molécule sont la cause de l'échec de 60% des médicaments en phase clinique.Une optimisation multi paramètres utilisant des méthodes d'apprentissage automatique peut être utilisée pour choisir une molécule optimisée qui sera soumise à des études plus approfondies et qui permettra d'éviter que la molécule ne devienne un médicament.
à soumettre à des études plus approfondies et d'éviter tout échec en phase clinique.
![Prédiction des propriétés d'une molécule de médicament.](https://fr.vikidia.org/wiki/Mol%C3%A9cule#/media/Fichier:Glucose.PNG)
# I. Modèle 
main.py, fait partie du projet MyFlaskApp et est responsable de l'entraînement et de l'exécution de deux modèles différents (Modèle1 Modèle 2et Modèle3) pour une application d'apprentissage automatique. Ce README donne un aperçu du script et comment l'utiliser.
Objectif
Le script a les objectifs suivants :

## 1. Entraînement de Modèle1 :

-Charge un ensemble de données à partir d'un fichier CSV.
-Extrait des caractéristiques à partir de chaînes SMILES en utilisant RDKit.
-Divise l'ensemble de données en ensembles d'entraînement, de validation et de test.
-Définit et entraîne Modèle1, qui est un modèle de réseau neuronal.
-Évalue le Modèle1 entraîné.
-Sauvegarde le Modèle1 entraîné dans un fichier.
Modèle 1 :

model1 = keras.Sequential([
    layers.Input(shape=(2048,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
Ce modèle est entrainé sur myflaskapp/data/single_dataset.csv
Entrèe : Le modèle prend en entrée un vecteur de forme (2048,)(les caractéristiques extraites de chaque molécule ont une longueur de 2048) . Ces caractéristiques sont extraites à l'aide de la méthode Morgan Fingerprint, qui est  utilisée pour la représentation des molécules.

 Le modèle utilise une architecture simple avec une couche dense de 64 unités suivie d'une couche de sortie avec une seule unité (car il s'agit d'une tâche de classification binaire: la prédiction d'une propriété binaire). La fonction d'activation utilisée dans la couche dense est ReLU  pour introduire de la non-linéarité, et la dernière couche utilise une activation sigmoïde pour la classification binaire.

Le modèle est compilé avec l'optimiseur Adam et la perte binaire_crossentropy. Cela indique que le modèle est formé pour la classification binaire en minimisant la perte de perte logistique.
## 2. Entraînement de Modèle2 :

-Convertit les chaînes SMILES en séquences d'entrée encodées en one-hot.
-Divise l'ensemble de données en ensembles d'entraînement, de validation et de test.
-Définit et entraîne Modèle2, qui est un modèle basé sur LSTM.
-Évalue le Modèle2 entraîné.
-Sauvegarde le Modèle2 entraîné dans un fichier.
-Exécution de l'Application Flask :
Le script est exécuté en tant que partie de l'application Flask, rendant les modèles entraînés disponibles pour des prédictions via des points d'API.
Modèle 2 :
Ce modèle est entrainé sur myflaskapp/data/single_dataset.csv
model2 = keras.Sequential([
    layers.Input(shape=(YOUR_INPUT_SHAPE, YOUR_VOCAB_SIZE)),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

Input Shape : Le modèle prend en entrée une séquence de forme (50, 30), ce qui suggère que le modèle est conçu pour prendre en compte la séquence de caractères (SMILES) des molécules. 

 Le modèle utilise une couche LSTM (Long Short-Term Memory), une couche récurrente, pour prendre en compte les dépendances séquentielles dans la représentation des molécules. La couche LSTM a 64 unités, ce qui peut capturer des motifs complexes dans les séquences. La dernière couche est similaire au modèle 1, avec une seule unité d'activation sigmoïde pour la classification binaire.

 Le modèle est également compilé pour la classification binaire avec l'optimiseur Adam et la perte binaire_crossentropy.
 ## 3. Entraînement de Modèle3 :
 Modèle 3 :


model3 = keras.Sequential([
    layers.Input(shape=(2048,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

Ce modèle est entrainé sur myflaskapp/data/multi_dataset.csv
Chaque modèle dans l'ensemble "Modèle 3" suit une architecture similaire à celle du "Modèle 1", avec une seule unité de sortie utilisant une activation sigmoïde. Cela signifie que chaque modèle génère une prédiction binaire indépendante pour sa propriété respective.
Le modèle 3  est entraîné pour chaque propriété de molécule individuelle (P1, P2, etc.), ce qui signifie qu'il y a un modèle distinct pour chaque propriété.Le modèle est compilé pour la classification binaire, ce qui suggère que chaque modèle de propriété est formé pour prédire une propriété binaire spécifique.

Justification globale :

Les choix de ces modèles sont adaptés aux tâches de prédiction de propriétés de molécules de médicaments. Le modèle 1 capture les caractéristiques moléculaires globales, tandis que le modèle 2 prend en compte les dépendances séquentielles dans la structure des molécules en utilisant une couche LSTM. Le modèle 3 est répété pour chaque propriété de molécule, ce qui permet de former un modèle distinct pour chaque propriété. Ces choix de modèles dépendent des caractéristiques de jeu de données et des tâches de prédiction spécifiques. 

## 4.Utilisation: 
Pour utiliser ce script, suivez ces étapes :
Préparation des Données :
Préparez l' ensemble de données dans un fichier CSV (par exemple, dataset_single.csv).
Assurez-vous que le fichier CSV contient les colonnes nécessaires (par exemple, 'smiles' et 'P1').
Entraînement de Modèle1 :
Exécutez le script pour entraîner Modèle1.
Modifiez les hyperparamètres ou l'architecture du modèle selon vos besoins dans le script.
Entraînement de Modèle2 :
Entraînez Modèle2 en utilisant des séquences encodées en one-hot de chaînes SMILES.
Modifiez les hyperparamètres ou l'architecture du modèle selon vos besoins dans le script.

## 5.Evaluation: 
Pour déterminer le meilleur modèle parmi les trois (Modèle 1, Modèle 2 et Modèle 3), nous pouvons nous baser sur plusieurs métriques de performance, telles que la perte de test et la précision. Voici les résultats des métriques pour chaque modèle :
Modèle 1:

Test loss: 0.9015
Test accuracy: 0.7640
Modèle 2:

Test loss: 0.4660
Test accuracy: 0.7907
Modèle 3:

Test loss pour P1: 0.9283
Test accuracy pour P1: 0.7747
Test loss pour P2: 0.7563
Test accuracy pour P2: 0.7720
Test loss pour P3: 0.7496
Test accuracy pour P3: 0.7947
Test loss pour P4: 0.8751
Test accuracy pour P4: 0.7707
Test loss pour P5: 0.6514
Test accuracy pour P5: 0.8293
Test loss pour P6: 0.8444
Test accuracy pour P6: 0.8040
Test loss pour P7: 0.8437
Test accuracy pour P7: 0.7693
Test loss pour P8: 0.7224
Test accuracy pour P8: 0.8053
Test loss pour P9: 0.7178
Test accuracy pour P9: 0.8213
Le choix du meilleur modèle dépendra de l'importance relative de la précision par rapport à la perte:
Si on privilégie la précision, alors Modèle 3 pour la Propriété P5 est le meilleur avec une précision de 0.8293, suivie de Modèle 2 avec une précision de 0.7907. Modèle 3 pour P5 a la meilleure précision globale.

Si on accorde de l'importance à la perte de test (plus faible est meilleure), Modèle 2 a la plus faible perte de test générale, suivie de Modèle 3 pour P5. Cela signifie que Modèle 2 produit des prédictions globalement plus proches des étiquettes de test.

Si on cherche un équilibre entre la précision et la perte de test, alors Modèle 3 pour P5 semble être un choix solide, car il a la meilleure précision tout en maintenant une perte de test raisonnable.

# II. Exécution de l'Application Flask :

Le script fait partie de l'application Flask. Exécutez l'application Flask pour fournir des points d'API pour effectuer des prédictions à l'aide des modèles entraînés
python app.py 

Accédez à l'application Flask à http://localhost:5000.
Informations Supplémentaires
Le script suppose une forme d'entrée spécifique et une taille de vocabulaire pour Modèle2. Modifiez ces constantes (50 et 30) en fonction de votre ensemble de données.
Les modèles entraînés sont sauvegardés dans des fichiers nommés 'model1.h5' , 'model2.h5' et 'model3.h5' . Ces fichiers peuvent être chargés et utilisés dans l'application Flask.
Assurez-vous d'avoir toutes les dépendances nécessaires installées pour exécuter le script avec succès, y compris  TensorFlow et RDKit.
MyFlaskApp - Application Flask pour les Prédictions Moléculaires
Le script app.py fait partie du projet MyFlaskApp et est responsable de l'application Flask pour effectuer des prédictions moléculaires en utilisant 3 modèles de machine learning (Modèle1  Modèle2 et Modèle 3). 

## 1.Objectif
L'objectif du script app.py est de fournir une interface web permettant aux utilisateurs de soumettre des chaînes SMILES de molécules et d'obtenir des prédictions à l'aide de deux modèles de machine learning (Modèle1 et Modèle2,Modèle 3). Les prédictions sont renvoyées sous forme de réponses JSON.

## 2. Fonctionnement
Le script fonctionne de la manière suivante :
###Chargement des Modèles de Machine Learning :
Les modèles de machine learning, Modèle1 Modèle 23et Modèle2, sont chargés à partir des fichiers 'model1.h5'  'model2.h5'et 'model3.h5'.
###Extraction de Caractéristiques SMILES :
Une fonction est définie pour extraire des caractéristiques à partir de chaînes SMILES en utilisant RDKit. Ces caractéristiques sont nécessaires pour les prédictions.
###Prétraitement des Chaînes SMILES :
Une fonction est définie pour prétraiter les chaînes SMILES en encodant en one-hot. Cela est nécessaire pour Modèle2.
###Fonctions de Prédiction :
3 fonctions, make_prediction1 make_prediction2, et make_prediction3, sont définies pour faire des prédictions en utilisant Modèle1 Modèle 2 et Modèle3 respectivement. Vous devez personnaliser ces fonctions selon vos besoins.
###Endpoints Flask :
L'application Flask définit deux endpoints :
/ : Renvoie la page d'accueil avec un formulaire pour soumettre des chaînes SMILES.
/predict : Accepte les chaînes SMILES soumises en tant que données JSON, effectue des prédictions en utilisant les modèles, et renvoie les prédictions sous forme de réponses JSON.


# III. MyFlaskApp - Packaging avec setup.py

Ce projet a été structuré pour être installable via setup.py, ce qui permet d'associer des commandes spécifiques au projet.
Installation avec setup.py :
Pour installer l'application, exécutez la commande suivante à partir du répertoire du projet :

Cela installera l'application localement.
##Commandes
Le setup.py inclut des commandes personnalisées pour effectuer diverses actions liées à l'application Flask.
Entraînement du Modèle
Pour entraîner un modèle, utilisez la commande suivante :
servier train <vos arguments>
Évaluation du Modèle
Pour évaluer un modèle, utilisez la commande suivante :
servier evaluate <vos arguments>
Installation
```
git clone https://github.com/rabebkaabi/RabebKaabi_Servier
cd RabebKaabi_Servier
pip install .
```
Prédiction
Pour effectuer des prédictions, utilisez la commande suivante :
servier predict <vos arguments>

# IV. MyFlaskApp - Déploiement avec Docker
Ce projet est une application Flask qui utilise Docker pour faciliter le déploiement. Il inclut un modèle de machine learning, un service Web Flask, et un Dockerfile pour créer une image Docker.
Contenu du Projet
Le projet est structuré comme suit :
app.py: Le code de l'application Flask.
requirements.txt: Liste des dépendances Python nécessaires pour l'application.
Dockerfile: Configuration Docker pour créer l'image du conteneur.
templates/: Répertoire contenant les modèles HTML pour l'interface utilisateur.
models/: Répertoire contenant les modèles de machine learning.
dataset/: Répertoire destiné à stocker les données d'entrée
Installation et Utilisation
Pour exécuter cette application, suivez ces étapes :
Cloner le projet :
```
git clone https://github.com/rabebkaabi/RabebKaabi_Servier
cd myflaskapp/myflaskapp
```
Rabeb Kaabi 
Construction de l'image Docker :
Assurez-vous que Docker Desktop est installé et fonctionne sur votre machine. Ensuite, exécutez la commande suivante pour construire l'image Docker :
docker build -t myflaskapp .
Exécution du conteneur Docker :
Une fois l'image Docker créée, vous pouvez exécuter un conteneur Docker à partir de l'image :
docker run -p 5000:5000 -v /chemin/vers/votre/dataset:/app/dataset myflaskapp
