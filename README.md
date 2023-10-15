# RabebKaabi_Servier


4.MyFlaskApp - Déploiement avec Docker
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
git clone https://github.com/rabebkaabi/RabebKaabi_Servier/myflaskapp-docker.git
cd myflaskapp-docker
Construction de l'image Docker :
Assurez-vous que Docker Desktop est installé et fonctionne sur votre machine. Ensuite, exécutez la commande suivante pour construire l'image Docker :
docker build -t myflaskapp .
Exécution du conteneur Docker :
Une fois l'image Docker créée, vous pouvez exécuter un conteneur Docker à partir de l'image :
docker run -p 5000:5000 -v /chemin/vers/votre/dataset:/app/dataset myflaskapp
