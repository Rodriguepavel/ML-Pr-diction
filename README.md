# Prédiction des fins de contrat de travail en entreprise — Une approche par Machine Learning

Projet académique de Machine Learning réalisé dans le cadre du cursus **Ingénieur Statisticien Économiste (ISE 2)** à l'[ISSEA](http://www.issea-cemac.org) (Institut Sous-régional de Statistique et d'Économie Appliquée, CEMAC), Yaoundé, Cameroun — Novembre 2024.

## Table des matières

- [Contexte et objectifs](#contexte-et-objectifs)
- [Jeu de données](#jeu-de-données)
- [Méthodologie](#méthodologie)
- [Résultats](#résultats)
- [Facteurs déterminants de l'attrition](#facteurs-déterminants-de-lattrition)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation et utilisation](#installation-et-utilisation)
- [Stack technique](#stack-technique)
- [Équipe](#équipe)
- [Références](#références)

## Contexte et objectifs

Dans un environnement économique de plus en plus concurrentiel, la fidélisation des employés est devenue un enjeu stratégique pour les entreprises : le départ d'un collaborateur qualifié entraîne à la fois une perte de compétences et des coûts de recrutement/formation significatifs.

Ce projet vise à construire un modèle prédictif de l'**attrition** (démission ou fin de contrat) des employés d'une entreprise, afin de :

- déterminer les caractéristiques d'un employé ayant l'intention de démissionner ;
- identifier les facteurs explicatifs les plus influents sur la décision de départ ;
- fournir aux équipes RH un outil d'aide à la décision pour cibler des actions de fidélisation.

## Jeu de données

- **1 470 employés**, avec **~35 variables** décrivant leur profil socio-démographique, professionnel et relationnel (âge, revenu, département, ancienneté, satisfaction au travail, fréquence des voyages professionnels, etc.).
- Variable cible : `Attrition` (`Yes` / `No`), fortement déséquilibrée — **83,9 %** des employés sont toujours en poste contre **16,1 %** ayant quitté l'entreprise (237 employés).
- Données quasi complètes : seules quelques valeurs manquantes sur la distance domicile–travail, imputées par la médiane (distribution étalée à droite).

## Méthodologie

### 1. Analyse exploratoire des données (EDA)

- Statistiques descriptives et visualisations (répartition de l'attrition, distribution des revenus par statut d'attrition, etc.).
- Mise en évidence de facteurs associés à un taux de départ plus élevé : département R&D, sexe masculin, faible satisfaction au travail, poste de technicien de laboratoire / chercheur scientifique, formation en sciences de la vie, faible fréquence de voyages professionnels, forte évaluation de performance.

### 2. Préparation des données

- Suppression des variables fortement corrélées (seuil > 0,7) identifiées via la matrice de corrélation : `Revenu-mensuel`/`TotalWorkingYears`, `YearsAtCompany`/`YearsInCurrentRole`, `YearsAtCompany`/`YearsWithCurrManager`, `YearsInCurrentRole`/`YearsWithCurrManager`.
- Sélection de variables par tests statistiques (ANOVA pour les variables quantitatives, Khi-2 pour les variables qualitatives) — suppression des variables faiblement liées à l'attrition (`BusinessTravel`, `Department`, `EnvironmentSatisfaction`, `JobInvolvement`, `JobSatisfaction`, `WorkLifeBalance`, `EducationField`, `MaritalStatus`).
- Traitement des valeurs aberrantes.
- Encodage : binarisation de la cible (`0`/`1`), one-hot encoding des variables catégorielles.

### 3. Traitement du déséquilibre de classes

Le fort déséquilibre de la variable cible (16,1 % de positifs) nécessite un rééquilibrage avant modélisation, sous peine de biaiser les modèles vers la classe majoritaire. Méthode retenue : **combinaison de sous-échantillonnage aléatoire et de SMOTE (Synthetic Minority Oversampling Technique)** — une approche hybride qui réduit la classe majoritaire puis synthétise de nouveaux exemples de la classe minoritaire, limitant à la fois le temps de calcul et le risque de surapprentissage.

### 4. Modélisation

Trois algorithmes de classification supervisée ont été comparés :

- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **Support Vector Machine (SVM)**

Split entraînement/test : 80 % / 20 %.

## Résultats

### Sur données déséquilibrées (baseline)

| Modèle | Accuracy | Recall | Precision | F1-score |
|---|---|---|---|---|
| Random Forest | 0.82 | 0.82 | 0.83 | 0.82 |
| KNN | 0.73 | 0.73 | 0.78 | 0.75 |

### Après rééquilibrage (sous-échantillonnage + SMOTE) et normalisation

| Modèle | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| KNN | 0.9076 | 0.8507 | 0.9869 | 0.9138 |
| **Random Forest** | **0.9190** | 0.9476 | 0.8856 | **0.9155** |
| SVM | 0.9173 | 0.9740 | 0.8562 | 0.9113 |

Le **Random Forest** offre le meilleur compromis global (meilleur F1-score) et a été retenu comme modèle final, avec une précision de l'ordre de 85–88 % selon la méthode de ré-échantillonnage. Il a ensuite été intégré à une interface simple permettant de saisir les informations d'un employé pour obtenir une prédiction d'attrition.

## Facteurs déterminants de l'attrition

L'analyse d'importance des variables (Random Forest) fait ressortir deux jeux de facteurs clés selon la méthode d'interprétation utilisée :

- **Méthode 1** : niveau de stock-options, revenu mensuel, ancienneté dans l'entreprise, âge, statut marital (marié).
- **Méthode 2** : fréquence des voyages d'affaires, âge, taux journalier, nombre total d'années travaillées, évaluation de performance.

Ces résultats confirment et enrichissent la littérature existante (Herzberg, Barriball & Farrell, Cameron, etc.) en mettant en évidence l'importance de la rémunération, de la mobilité professionnelle et de la charge de travail annuelle dans la décision de départ.

## Structure du dépôt

```
.
├── data/
│   └── employee_attrition.csv        # jeu de données (1 470 employés)
├── notebooks/
│   ├── 01_exploration.ipynb          # analyse exploratoire des données
│   ├── 02_preprocessing.ipynb        # nettoyage, sélection de variables, encodage
│   ├── 03_resampling.ipynb           # traitement du déséquilibre (SMOTE)
│   └── 04_modelisation.ipynb         # entraînement et évaluation des modèles
├── app/
│   └── interface_prediction.py       # interface de prédiction (saisie manuelle)
├── rapport/
│   └── Rapport.pdf                   # rapport complet du projet
├── requirements.txt
└── README.md
```

> À adapter selon l'organisation réelle du dépôt.

## Installation et utilisation

```bash
# Cloner le dépôt
git clone https://github.com/<votre-organisation>/employee-attrition-prediction.git
cd employee-attrition-prediction

# Créer un environnement virtuel et installer les dépendances
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt

# Lancer l'exploration / la modélisation
jupyter notebook notebooks/

# Lancer l'interface de prédiction
python app/interface_prediction.py
```

## Stack technique

- **Python** (pandas, numpy)
- **scikit-learn** (KNN, Random Forest, SVM, métriques d'évaluation)
- **imbalanced-learn** (SMOTE, sous-échantillonnage)
- **matplotlib / seaborn / plotly** (visualisations)
- **scipy / statsmodels** (tests ANOVA, Khi-2)

## Équipe

Projet réalisé par le **Groupe 14**, élèves Ingénieurs Statisticiens Économistes en 2ᵉ année à l'ISSEA :

- AWADJOU DEUFACK Rodrigue Pavel
- KENGNE Bienvenu Landry
- NGONO MVOGO Franck
- NJIFON MOUNCHINGAM Seidou Ayatou

## Références

- Herzberg, F. (2003). *One More Time: How Do You Motivate Employees?* Harvard Business Review.
- Herzberg, F. (1959). *The Motivation to Work*. New York: John Wiley & Sons.
- Barriball, C. & Farrell, R. (2007). *Impact of job satisfaction components on intent to leave and turnover for hospital-based nurses: A review of the research literature.* International Journal of Nursing Studies.
- Cameron, W. (2011). *Developing Management Skills* (8th ed.). Pearson, Prentice Hall.
- Pratt, M. C. (2021). *Motivation in a Business Company Using Technology-Based Communication.* In Artificial Intelligence in Industry 4.0, Studies in Computational Intelligence.
- Alao, D. A. (2013). *Analyzing employee attrition using decision tree algorithms.* Computing, Information Systems, Development Informatics and Allied Research Journal.
- Mveng Minkoulou, G. Y. (2006). *Fidélisation du personnel et performance de l'entreprise : une application au personnel d'encadrement de Guinness Cameroun s.a.*

---

*Rapport complet disponible dans [`rapport/Rapport.pdf`](./rapport/Rapport.pdf).*
