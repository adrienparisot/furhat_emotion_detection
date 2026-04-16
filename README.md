# Détection d'émotions avec furhat

Ce projet reprend différents projets de reconnaissance d'émotions (happy, fear, sad, disgust, surprise...) avec furhat. Il a pour but de repérer les émotions réalisées par l'utilisateur devant la caméra intégrée du robot. Celui-ci doit pouvoir énoncer quelle émotion a été réalisé et la reproduire. Le modèle initial fine tuné est DenseNet-121
https://docs.furhat.io/remote-api/.



### Travail repris

Nous avons repris le travail des TPE de l'année précédente

- https://github.com/fab-toc/FER_Furhat
- https://github.com/GuidonAntoine/facial-expression-recognition-4Classes.git
- https://github.com/cosmiclf/furhat_react_to_emotion


## Expressions reconnues 

    😠 Colère (Angry) - LED rouge, expression de colère
    😨 Peur (Fear) - LED violette, expression de peur
    😊 Joie (Happy) - LED jaune, grand sourire
    😢 Tristesse (Sad) - LED bleue, expression triste
    😲 Surprise (Surprise) - LED verte, expression de surprise
    😐 Neutre (neutral) - pas de LED, particulier car pas fait avec le modèle 


## Installation : Tutoriel 

```bash
# Cloner le repository
git clone https://github.com/AdrienParisot/furhat_emotion_detection
cd furhat_emotion_detection

# Créer un environnement virtuel
python -m venv env

# Activer l'environnement
source env/bin/activate  # Linux / macOS

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation
Afin que le robot reconnaisse l'expression de l'utilisateur il faut lancer le programme `detection_emotion.py` et regarder la caméra se trouvant en-dessous de la tête du robot Furhat. Il est important d'exagérer l'émotion. 

```bash
python detection_emotion.py
```


Le programme `furhat_ollama.py` permet de parler au robot et de lui demander de faire la détection d'émotion en disant une phrase qui contient le mot émotion.
```bash
python furhat_ollama.py
```

## Structure du projet

```
furhat_emotion_detection/
├── 📁 Dense5classes/          # Les modèles
│   ├── best2_densenet121_acc0.7304.pth            # Modèle entraîné sur la base de données affectnet_fer
    ├── best2_finetuned_acc0.9337.pth              # Modèle entraîné sur la base de données dataset_5classes à partir du modèle best2_densenet121_acc0.7304.pth
    ├── best_densenet121_acc0.7600.pth             # Modèle entraîné sur la base de données FER-2013
    ├── best_finetuned_acc0.9587.pth               # Modèle entrainé sur la base de données dataset_5classes à partir du modèle best_densenet121_acc0.7600.pth
    ├── best_finetuned_acc0.9668.pth               # Modèle entraîné sur la base de données dataset_5classes  
    └── infos.txt                                  # Infos sur les différents modèles
├── 📁 figure/                 # Fichier contenant les courbes
    ├── finetuning.png         # Courbe faite sur le training du modèle best2_finetuned_acc0.9337.pth
    └── pretraining.png        # Courbe faite sur le training du modèle best2_densenet121_acc0.7304.pth
├── 📁 photo_test/             # Contient les photos pour tester le finetuning
├── 📁 prompts/                # Contient les prompts pour parler au Furhat
├── 📁 dataset/
    ├── 📁 affectnet_fer       # Base de donnée mélangeant AffectNet et FER-2013
    ├── 📁 dataset_5classes    # Base de donnée mélangeant les photos prises cette année (2026) et celles prises les années précédentes sur les 5 classes que l'on utilise
    └── 📁 dataset_complet     # Base de donnée mélangeant les photos prises cette année (2026) et celles prises les années précédentes sur les 7 classes disponibles
├── DenseNet_5classe.ipynb                         # Fichier permettant de faire l'entraienment sur la base de données affectnet_fer 
├── detection_emotion.py                           # Programme pour lancer la detection des émotions avec Furhat en continue
├── fine_tuning.py                                 # Programme pour lancer le fine-tunning d'un modèle à partir de la base de données dataset_5classes 
├── furhat_ollama.py                               # Programme pour parler au robot et lui demander la détection d'émotion en disant une phrase contenant le mot "émotion"   
├── photo.py                                       # Permet de prendre des photos avec la caméra du Furhat
├── testfinetuned.py                               # Tester une le fine-tuning à partir d'une photo contenue dans photo_test
├── requirements.txt                               # Packages utiles au lancement des différents programmes       
└── README.md                                      # Cette documentation
```

## Modèle : DenseNet121

* Base : pré-entraîné sur ImageNet
* Fine-tuning : uniquement à partir du bloc denseblock3
* Sortie personnalisée : 5 classes (angry, happy, surprise, sad, fear)


## Performances

### Métriques Typiques

| Dataset                     | Modèle         | Précision |
| --------------------------- | -------------- | --------- |
| affectnet_fer               | DenseNet121    | ~73%      |
| dataset_5classes            | DenseNet121    | ~93%      |


