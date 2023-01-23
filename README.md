
##Pour script Training.ipynb

Ce code est un script d'entraînement pour un modèle de segmentation d'images médicales basé sur UNet. Il utilise la bibliothèque Monai pour les réseaux de neurones, les pertes et les normalisations.

Il commence par définir des variables d'environnement et importer les modules nécessaires.
 Ensuite, il définit une liste de MRI (1 à 6 et 9) pour lesquels il souhaite entraîner des modèles. 
Pour chaque itération de la boucle, il crée un répertoire pour stocker le modèle entraîné, charge les données d'entraînement et de test à partir d'un emplacement spécifié, 
Il instancie un modèle UNet en utilisant les paramètres spécifiés, définit une fonction de perte (DiceLoss) et un optimiseur (Adam). 
Puis appelle la fonction train pour entraîner le modèle pendant 200 époques
. Le modèle entraîné est enregistré dans le répertoire créé précédemment.


##Pour script Test_et_visualisation.ipynb


Ce code est utilisé pour charger les données d'entraînement et de test, afficher les courbes de performance d'un modèle entraîné et effectuer une inférence glissante sur un modèle de segmentation d'images médicales UNet. 
Il utilise la bibliothèque Monai pour les réseaux de neurones, les transforms de données et les inférences.

Il commence par importer les modules nécessaires, puis définit les chemins vers les données d'entraînement et de test. 
Il charge les courbes de performance du modèle entraîné (perte d'entraînement, métrique d'entraînement, perte de test et métrique de test) et les affiche dans une figure. 
Il utilise ensuite la fonction glob pour charger les volumes et les segmentations d'entraînement et de test. 

##Initialisation des données 

Mettre les données dans cc359_preprocessed , diviser les données en 6 dossiers , chacun contenant les donneés de chaque IRM
Ensuite , diviser ces fichiers en données en entrainement/test ainsi qu'en segmenatation et volumes 
Voir image Organisation-fichier.JPG pour une idée de la disposition
