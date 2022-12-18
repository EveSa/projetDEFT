Ce projet répond à la tâche 3 du Défi Fouille de texte de 2009 qui étudie la classification d'opinion multilingue via la classification des prises de paroles au parlement européen par groupe parlementaire.

# Les dossiers
- /bin :
    contient les scripts nécessaires à la classification des prises de paroles
- /data :
    contient les fichiers pré-traités des prises de paroles en français, anglais et italiens transformés depuis le dossier /files. 
- /files : Contient les fichiers bruts à télécharger sur https://deft.limsi.fr/ 
- /model :
    contient les meilleurs modèles entraînés lors des expériences dans les dossiers suivant
    - /DecisionTree
    - /LinearSVC
    - /LogisticRegression
    - /RandomForest
    - /SGDClassifier
    - /SVCClassification
- /rapport :
    contient les matrices de confusions des test et le rapport au format pdf

# Installation

L'utilisation des scripts nécéssite l'installation des bibliothèques spécifiées dans `requirements.txt`

# Paramètres

Pour utiliser les scripts, il suffit de modifier les paramètres selon vos préférences dans `bin/main.py`. Les paramètres modifiables sont les suivants :
- la langue `lang` parmi :
    - 'en'
    - 'fr'
    - 'it'
- le prétraitement `name` parmi :
    - 'wthtpunct' pour enlever la ponctutation
    - 'lemmatized' pour lematiser les données
    - 'balanced' pour rééquilibrer les données
- la profondeur maximale `max_depth` si le modèle testé en requiert une :
    - un int
- le nom du modèle `model_name` à tester parmi :
    - 'decision_tree'
    - 'random_forest'
    - 'logistic_regression'
    - 'linearsvc_classification'
    - 'svc_classification'
    - 'SGD_classification'

# Utilisation

Lancer la commande `bin/main.py` depuis le dossier contenant le projet dans le terminal

____

La fonction utilisant le modèle Bert n'a pas été utilisée du fait des mauvais résulats trouvé lors de l'entraînement et de la difficulté de l'implémentation mais a été laissée pour montrer la tentative.
