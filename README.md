# RabebKaabi_Servier
La prédiction des propriétés d'une molécule de médicament joue un rôle important dans le processus de conception de médicaments. Les propriétés de la molécule sont la cause de l'échec de 60% des médicaments en phase clinique.Une optimisation multi paramètres utilisant des méthodes d'apprentissage automatique peut être utilisée pour choisir une molécule optimisée qui sera soumise à des études plus approfondies et qui permettra d'éviter que la molécule ne devienne un médicament.
à soumettre à des études plus approfondies et d'éviter tout échec en phase clinique.
![Prédiction des propriétés d'une molécule de médicament.](https://fr.vikidia.org/wiki/Mol%C3%A9cule#/media/Fichier:Glucose.PNG)
# I. Modèle 
main.py, fait partie du projet MyFlaskApp et est responsable de l'entraînement et de l'exécution de deux modèles différents (Modèle1 et Modèle2) pour une application d'apprentissage automatique. Ce README donne un aperçu du script et comment l'utiliser.
```
python main.py
```
Objectif
Le script a les objectifs suivants :

## 1. Entraînement de Modèle1 :

-Charge un ensemble de données à partir d'un fichier CSV.
-Extrait des caractéristiques à partir de chaînes SMILES en utilisant RDKit.
-Divise l'ensemble de données en ensembles d'entraînement, de validation et de test.
-Définit et entraîne Modèle1, qui est un modèle de réseau neuronal.
-Évalue le Modèle1 entraîné.
-Sauvegarde le Modèle1 entraîné dans un fichier.

## 2. Entraînement de Modèle2 :

-Convertit les chaînes SMILES en séquences d'entrée encodées en one-hot.
-Divise l'ensemble de données en ensembles d'entraînement, de validation et de test.
-Définit et entraîne Modèle2, qui est un modèle basé sur LSTM.
-Évalue le Modèle2 entraîné.
-Sauvegarde le Modèle2 entraîné dans un fichier.
-Exécution de l'Application Flask :
Le script est exécuté en tant que partie de l'application Flask, rendant les modèles entraînés disponibles pour des prédictions via des points d'API.
## 3.Utilisation: 
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

# II. Exécution de l'Application Flask :
```
python app.py
```
Le script fait partie de l'application Flask. Exécutez l'application Flask pour fournir des points d'API pour effectuer des prédictions à l'aide des modèles entraînés.
Accédez à l'application Flask à http://localhost:5000.
Informations Supplémentaires
Le script suppose une forme d'entrée spécifique et une taille de vocabulaire pour Modèle2. Modifiez ces constantes (YOUR_INPUT_SHAPE=50 et YOUR_VOCAB_SIZE=30) en fonction de votre ensemble de données.
Les modèles entraînés sont sauvegardés dans des fichiers nommés 'model1.h5' et 'model2.h5'. Ces fichiers peuvent être chargés et utilisés dans l'application Flask.
Assurez-vous d'avoir toutes les dépendances nécessaires installées pour exécuter le script avec succès, y compris RDKit, TensorFlow et RDKit.
MyFlaskApp - Application Flask pour les Prédictions Moléculaires
Le script app.py fait partie du projet MyFlaskApp et est responsable de l'application Flask pour effectuer des prédictions moléculaires en utilisant deux modèles de machine learning (Modèle1 et Modèle2). 

## 1.Objectif
L'objectif du script app.py est de fournir une interface web permettant aux utilisateurs de soumettre des chaînes SMILES de molécules et d'obtenir des prédictions à l'aide de deux modèles de machine learning (Modèle1 et Modèle2). Les prédictions sont renvoyées sous forme de réponses JSON.

## 2. Fonctionnement
Le script fonctionne de la manière suivante :
###Chargement des Modèles de Machine Learning :
Les modèles de machine learning, Modèle1 et Modèle2, sont chargés à partir des fichiers 'model1.h5' et 'model2.h5'.
###Extraction de Caractéristiques SMILES :
Une fonction est définie pour extraire des caractéristiques à partir de chaînes SMILES en utilisant RDKit. Ces caractéristiques sont nécessaires pour les prédictions.
###Prétraitement des Chaînes SMILES :
Une fonction est définie pour prétraiter les chaînes SMILES en encodant en one-hot. Cela est nécessaire pour Modèle2.
###Fonctions de Prédiction :
Deux fonctions, make_prediction1 et make_prediction2, sont définies pour faire des prédictions en utilisant Modèle1 et Modèle2 respectivement. Vous devez personnaliser ces fonctions selon vos besoins.
###Endpoints Flask :
L'application Flask définit deux endpoints :
/ : Renvoie la page d'accueil avec un formulaire pour soumettre des chaînes SMILES.
/predict : Accepte les chaînes SMILES soumises en tant que données JSON, effectue des prédictions en utilisant les modèles, et renvoie les prédictions sous forme de réponses JSON.
###Exécution de l'Application :
L'application est exécutée localement et peut être accédée à l'adresse http://localhost:5000 dans un navigateur.
###Utilisation
Pour utiliser cette application Flask, suivez ces étapes :
Chargement des Modèles :
Assurez-vous que les modèles de machine learning, Modèle1 et Modèle2, sont disponibles dans des fichiers 'model1.h5' et 'model2.h5'. Placez ces fichiers dans le répertoire models/ du projet.
Exécution de l'Application :
Exécutez le script app.py pour lancer l'application Flask.
Accédez à l'application à l'adresse http://localhost:5000 dans votre navigateur.
Soumission de Prédictions :
Sur la page d'accueil, saisissez une chaîne SMILES dans le formulaire et cliquez sur le bouton "Predict".
Les prédictions des deux modèles seront affichées sur la page.
Personnalisation
Personnalisez les fonctions make_prediction1 et make_prediction2 pour adapter les prédictions en fonction des besoins de votre modèle.
Personnalisez l'interface utilisateur HTML en modifiant le fichier de modèle index.html .
Personnalisez les routes ou ajoutez des fonctionnalités supplémentaires en fonction de vos besoins spécifiques.

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
git clone https://github.com/rabebkaabi/RabebKaabi_Servier/myflaskapp.git
cd myflaskapp
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
Crée requirements.txt :
```
pip freeze > requirements.txt
pip install -r requirements.txt
```
Création de dockrfile sans extension dans le meme path que l'application :
```
# Utilisez une image de base Python
FROM python:3.8

# Copiez le fichier requirements.txt dans le conteneur
COPY requirements.txt /app/

# Définissez le répertoire de travail
WORKDIR /app

# Installez les dépendances à partir du fichier requirements.txt
RUN pip install -r requirements.txt

# Copiez tout le code source de votre application dans le conteneur
COPY . /app/
```
Construction de l'image Docker :
Assurez-vous que Docker Desktop est installé et fonctionne sur votre machine. Ensuite, exécutez la commande suivante pour construire l'image Docker :
docker build -t servier
Exécution du conteneur Docker :
Une fois l'image Docker créée, vous pouvez exécuter un conteneur Docker à partir de l'image :
docker run -p 5000:5000 -v /chemin/vers/data:/app/data myflaskapp
