# FastAPI - Parcours Data Scientist

API complète pour les 5 TPs du parcours Data Scientist : Clean → EDA → Multivarié → ML Baseline → ML Avancé.

## Description

Ce projet implémente une API REST avec FastAPI permettant de réaliser l'ensemble du parcours d'un projet data science :

1. **TP1 - Nettoyage & Préparation** : Traitement des valeurs manquantes, doublons, outliers
2. **TP2 - EDA** : Statistiques descriptives et visualisations
3. **TP3 - Analyse Multivariée** : PCA et clustering K-Means
4. **TP4 - ML Baseline** : Entraînement et prédiction de modèles
5. **TP5 - ML Avancé** : Optimisation d'hyperparamètres et explicabilité

## Architecture

```
fastapi-data-scientist/
├── docker-compose.yml          # Configuration Docker Compose
├── Dockerfile                  # Image Docker de l'application
├── requirements.txt            # Dépendances Python
├── README.md                   # Ce fichier
├── app/
│   ├── main.py                # Point d'entrée FastAPI
│   ├── config.py              # Configuration globale
│   ├── routers/               # Endpoints par phase
│   │   ├── dataset.py         # Génération de datasets
│   │   ├── clean.py           # TP1 - Nettoyage
│   │   ├── eda.py             # TP2 - EDA
│   │   ├── mv.py              # TP3 - Analyse multivariée
│   │   ├── ml.py              # TP4 - ML Baseline
│   │   └── ml2.py             # TP5 - ML Avancé
│   ├── services/              # Logique métier
│   │   ├── dataset_generator.py
│   │   ├── clean_service.py
│   │   ├── eda_service.py
│   │   ├── mv_service.py
│   │   ├── ml_service.py
│   │   └── ml2_service.py
│   ├── schemas/               # Modèles Pydantic
│   │   ├── common.py
│   │   ├── clean.py
│   │   ├── eda.py
│   │   ├── mv.py
│   │   ├── ml.py
│   │   └── ml2.py
│   └── utils/
│       └── storage.py         # Gestion stockage modèles
├── models/                    # Modèles sauvegardés
└── data/                      # Datasets générés
```

## Installation et Démarrage

### Avec Docker (recommandé)

1. **Cloner le repository**
```bash
git clone https://github.com/LilRaphh/DatascienceSupDeVinci
cd fastapi-data-scientist
```

2. **Lancer l'application**
```bash
docker-compose up -d
```

3. **Accéder à la documentation interactive**
```
http://localhost:8000/docs
```

### Sans Docker

1. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Lancer l'application**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Utilisation

### Workflow Général

1. **Générer un dataset** pour une phase spécifique
2. **Appeler les endpoints** de la phase correspondante
3. **Réutiliser** le `dataset_id` ou `model_id` dans les requêtes suivantes

### Exemple : TP1 - Nettoyage

```bash
# 1. Générer un dataset pour la phase "clean"
curl -X POST "http://localhost:8000/dataset/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "phase": "clean",
    "seed": 42,
    "n": 1000
  }'

# Réponse : {"meta": {...}, "result": {"dataset_id": "clean_42_1000_abc123", ...}}

# 2. Générer un rapport de qualité
curl -X GET "http://localhost:8000/clean/report/clean_42_1000_abc123"

# 3. Apprendre un pipeline de nettoyage
curl -X POST "http://localhost:8000/clean/fit?dataset_id=clean_42_1000_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "impute_strategy": "mean",
    "outlier_strategy": "clip",
    "categorical_strategy": "one_hot"
  }'

# Réponse : {"result": {"cleaner_id": "cleaner_xyz789", ...}}

# 4. Appliquer le nettoyage
curl -X POST "http://localhost:8000/clean/transform?dataset_id=clean_42_1000_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "cleaner_id": "cleaner_xyz789"
  }'
```

### Exemple : TP2 - EDA

```bash
# 1. Générer un dataset EDA
curl -X POST "http://localhost:8000/dataset/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "phase": "eda",
    "seed": 42,
    "n": 1000
  }'

# 2. Statistiques descriptives
curl -X POST "http://localhost:8000/eda/summary?dataset_id=eda_42_1000_xyz"

# 3. Groupby avec agrégations
curl -X POST "http://localhost:8000/eda/groupby?dataset_id=eda_42_1000_xyz" \
  -H "Content-Type: application/json" \
  -d '{
    "by": "segment",
    "metrics": ["mean", "median"]
  }'

# 4. Matrice de corrélation
curl -X POST "http://localhost:8000/eda/correlation?dataset_id=eda_42_1000_xyz"

# 5. Générer des graphiques
curl -X POST "http://localhost:8000/eda/plots?dataset_id=eda_42_1000_xyz" \
  -H "Content-Type: application/json" \
  -d '{
    "plot_types": ["histogram", "boxplot"],
    "numeric_var": "income",
    "group_by": "segment"
  }'
```

### Exemple : TP3 - Analyse Multivariée

```bash
# 1. Générer dataset multivarié
curl -X POST "http://localhost:8000/dataset/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "phase": "mv",
    "seed": 42,
    "n": 1000
  }'

# 2. PCA
curl -X POST "http://localhost:8000/mv/pca/fit_transform?dataset_id=mv_42_1000_xyz" \
  -H "Content-Type: application/json" \
  -d '{
    "n_components": 3,
    "scale": true
  }'

# 3. Clustering K-Means
curl -X POST "http://localhost:8000/mv/cluster/kmeans?dataset_id=mv_42_1000_xyz" \
  -H "Content-Type: application/json" \
  -d '{
    "k": 3,
    "scale": true
  }'

# 4. Rapport interprétatif
curl -X GET "http://localhost:8000/mv/report/mv_42_1000_xyz"
```

### Exemple : TP4 - ML Baseline

```bash
# 1. Générer dataset ML
curl -X POST "http://localhost:8000/dataset/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "phase": "ml",
    "seed": 42,
    "n": 1000
  }'

# 2. Entraîner un modèle
curl -X POST "http://localhost:8000/ml/train?dataset_id=ml_42_1000_xyz" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "rf",
    "test_size": 0.2,
    "rf_n_estimators": 100
  }'

# Réponse : {"result": {"model_id": "rf_model123", ...}}

# 3. Consulter les métriques
curl -X GET "http://localhost:8000/ml/metrics/rf_model123"

# 4. Faire des prédictions
curl -X POST "http://localhost:8000/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "rf_model123",
    "data": [
      {"x1": 15.2, "x2": 22.5, "x3": 10.0, "x4": 5.5, "x5": 30.0, "x6": 12.0, "segment": "A"}
    ]
  }'

# 5. Informations du modèle
curl -X GET "http://localhost:8000/ml/model-info/rf_model123"
```

### Exemple : TP5 - ML Avancé

```bash
# 1. Utiliser le même dataset que TP4
# ou générer avec phase="ml2"

# 2. Optimiser les hyperparamètres
curl -X POST "http://localhost:8000/ml2/tune?dataset_id=ml_42_1000_xyz" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "rf",
    "search": "grid",
    "cv": 5,
    "scoring": "f1"
  }'

# Réponse : {"result": {"best_model_id": "rf_tuned_abc", "best_params": {...}}}

# 3. Importance des features (native)
curl -X GET "http://localhost:8000/ml2/feature-importance/rf_tuned_abc"

# 4. Importance par permutation
curl -X POST "http://localhost:8000/ml2/permutation-importance?dataset_id=ml_42_1000_xyz" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "rf_tuned_abc",
    "n_repeats": 10
  }'

# 5. Explication d'une instance
curl -X POST "http://localhost:8000/ml2/explain-instance" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "rf_tuned_abc",
    "instance": {
      "x1": 15.2,
      "x2": 22.5,
      "x3": 10.0,
      "x4": 5.5,
      "x5": 30.0,
      "x6": 12.0,
      "segment": "A"
    }
  }'
```

## Endpoints Disponibles

### Dataset (Commun)
- `POST /dataset/generate` - Générer un dataset pour une phase
- `GET /dataset/info/{dataset_id}` - Informations sur un dataset
- `GET /dataset/preview/{dataset_id}` - Aperçu des données

### TP1 - Clean
- `POST /clean/fit` - Apprendre un pipeline de nettoyage
- `POST /clean/transform` - Appliquer le nettoyage
- `GET /clean/report/{dataset_id}` - Rapport de qualité

### TP2 - EDA
- `POST /eda/summary` - Statistiques descriptives
- `POST /eda/groupby` - Agrégations par groupe
- `POST /eda/correlation` - Matrice de corrélation
- `POST /eda/plots` - Génération de graphiques

### TP3 - MV
- `POST /mv/pca/fit_transform` - PCA
- `POST /mv/cluster/kmeans` - Clustering K-Means
- `GET /mv/report/{dataset_id}` - Rapport interprétatif

### TP4 - ML
- `POST /ml/train` - Entraîner un modèle
- `GET /ml/metrics/{model_id}` - Métriques de performance
- `POST /ml/predict` - Faire des prédictions
- `GET /ml/model-info/{model_id}` - Informations du modèle

### TP5 - ML2
- `POST /ml2/tune` - Optimiser les hyperparamètres
- `GET /ml2/feature-importance/{model_id}` - Importance native
- `POST /ml2/permutation-importance` - Importance par permutation
- `POST /ml2/explain-instance` - Explication locale

## Phases Disponibles

- `clean` : Données avec missing values, doublons, outliers, types cassés
- `eda` : Données pour analyse exploratoire
- `mv` : Données pour PCA et clustering
- `ml` / `ml2` : Données pour classification binaire

## Technologies Utilisées

- **FastAPI** : Framework web moderne et performant
- **Pydantic** : Validation de données et schémas
- **Pandas** : Manipulation de données
- **Scikit-learn** : Machine learning
- **Plotly** : Visualisations interactives
- **Docker** : Conteneurisation
- **Joblib** : Sérialisation des modèles

## Tests

Tous les endpoints sont testables via l'interface Swagger UI :
```
http://localhost:8000/docs
```

## Structure des Réponses

Toutes les réponses suivent le format standardisé :

```json
{
  "meta": {
    "dataset_id": "...",
    "timestamp": "2026-02-10T12:00:00",
    "schema_version": "1.0",
    "status": "success"
  },
  "result": {
    // Résultat principal
  },
  "report": {
    "message": "...",
    "warnings": [],
    "metrics": {}
  },
  "artifacts": {
    // Artefacts (graphiques, etc.)
  }
}
```

## Auteur - COLNOT Raphael

Projet réalisé dans le cadre du Parcours Data Scientist FastAPI.

## Licence

MIT
