# Tests et Vérification

## Tests Manuels via Swagger UI

### Test 1 : Vérification de l'API

1. Accédez à `http://localhost:8000/docs`
2. Testez l'endpoint `/health`
3. Vérifiez que la réponse est `{"status": "healthy"}`

### Test 2 : Génération de Dataset

1. Endpoint : `POST /dataset/generate`
2. Body :
```json
{
  "phase": "clean",
  "seed": 42,
  "n": 500
}
```
3. Vérifiez :
   - Status 200
   - Présence de `dataset_id` dans la réponse
   - `n_rows` = 500
   - `data_sample` contient 20 lignes

### Test 3 : TP1 - Nettoyage

**3.1 Rapport de qualité**
- Endpoint : `GET /clean/report/{dataset_id}`
- Vérifiez la présence de :
  - `n_duplicates` > 0
  - `missing_values` avec valeurs > 0
  - `outliers_count` avec valeurs > 0

**3.2 Fit du cleaner**
- Endpoint : `POST /clean/fit`
- Body :
```json
{
  "impute_strategy": "mean",
  "outlier_strategy": "clip",
  "categorical_strategy": "one_hot"
}
```
- Vérifiez :
  - `cleaner_id` retourné
  - `quality_before` rempli

**3.3 Transform**
- Endpoint : `POST /clean/transform`
- Body : `{"cleaner_id": "votre_cleaner_id"}`
- Vérifiez :
  - `n_rows_after` <= `n_rows_before`
  - `counters` avec valeurs cohérentes

### Test 4 : TP2 - EDA

Générez d'abord un dataset avec `phase="eda"`.

**4.1 Summary**
- Endpoint : `POST /eda/summary`
- Vérifiez :
  - Statistiques pour toutes les colonnes
  - `mean`, `std`, `min`, `max` présents pour numériques

**4.2 Groupby**
- Endpoint : `POST /eda/groupby`
- Body :
```json
{
  "by": "segment",
  "metrics": ["mean", "median"]
}
```
- Vérifiez 3 groupes (A, B, C)

**4.3 Correlation**
- Endpoint : `POST /eda/correlation`
- Vérifiez :
  - `correlation_matrix` est une matrice carrée
  - `top_correlations` contient des paires

**4.4 Plots**
- Endpoint : `POST /eda/plots`
- Body :
```json
{
  "plot_types": ["histogram"],
  "numeric_var": "income"
}
```
- Vérifiez :
  - `artifacts` contient "histogram"
  - Format JSON Plotly valide

### Test 5 : TP3 - Analyse Multivariée

Générez un dataset avec `phase="mv"`.

**5.1 PCA**
- Endpoint : `POST /mv/pca/fit_transform`
- Body :
```json
{
  "n_components": 2,
  "scale": true
}
```
- Vérifiez :
  - 2 composantes dans `components_info`
  - `total_variance_explained` > 0
  - `projection` contient PC1 et PC2

**5.2 Clustering**
- Endpoint : `POST /mv/cluster/kmeans`
- Body :
```json
{
  "k": 3,
  "scale": true
}
```
- Vérifiez :
  - 3 clusters dans `clusters_info`
  - `silhouette_score` entre -1 et 1
  - Somme des tailles = n_points

### Test 6 : TP4 - ML Baseline

Générez un dataset avec `phase="ml"`.

**6.1 Train**
- Endpoint : `POST /ml/train`
- Body :
```json
{
  "model_type": "logreg",
  "test_size": 0.2
}
```
- Vérifiez :
  - `model_id` retourné
  - `metrics_test.accuracy` entre 0 et 1
  - `metrics_test.f1_score` entre 0 et 1

**6.2 Metrics**
- Endpoint : `GET /ml/metrics/{model_id}`
- Vérifiez toutes les métriques présentes

**6.3 Predict**
- Endpoint : `POST /ml/predict`
- Body :
```json
{
  "model_id": "votre_model_id",
  "data": [
    {
      "x1": 15.0, "x2": 22.0, "x3": 10.0,
      "x4": 5.0, "x5": 30.0, "x6": 12.0,
      "segment": "A"
    }
  ]
}
```
- Vérifiez :
  - Prédiction 0 ou 1
  - Probabilités entre 0 et 1

### Test 7 : TP5 - ML Avancé

**7.1 Tune**
- Endpoint : `POST /ml2/tune`
- Body :
```json
{
  "model_type": "rf",
  "search": "grid",
  "cv": 3
}
```
- Vérifiez :
  - `best_model_id` retourné
  - `best_params` non vide
  - `cv_results_summary` contient 5 configs

**7.2 Feature Importance**
- Endpoint : `GET /ml2/feature-importance/{model_id}`
- Vérifiez :
  - Liste d'importances triées
  - `top_features` contient 5 features

**7.3 Explain Instance**
- Endpoint : `POST /ml2/explain-instance`
- Body :
```json
{
  "model_id": "votre_model_id",
  "instance": {
    "x1": 15.0, "x2": 22.0, "x3": 10.0,
    "x4": 5.0, "x5": 30.0, "x6": 12.0,
    "segment": "A"
  }
}
```
- Vérifiez :
  - `prediction` présent
  - `contributions` non vide
  - `explanation_summary` lisible

## Checklist de Vérification Complète

### Démarrage
- [ ] Docker Compose démarre sans erreur
- [ ] API accessible sur port 8000
- [ ] Swagger UI s'affiche correctement
- [ ] Endpoint `/health` répond "healthy"

### Génération de Datasets
- [ ] Phase "clean" génère données avec défauts
- [ ] Phase "eda" génère données cohérentes
- [ ] Phase "mv" génère 8 variables numériques
- [ ] Phase "ml" génère données avec target
- [ ] Même seed produit même dataset

### TP1 - Clean
- [ ] Rapport détecte tous les problèmes
- [ ] Cleaner s'entraîne sans erreur
- [ ] Transform réduit les problèmes
- [ ] Doublons supprimés correctement
- [ ] NA imputés correctement

### TP2 - EDA
- [ ] Summary calcule toutes les stats
- [ ] Groupby agrège correctement
- [ ] Corrélations calculées
- [ ] Graphiques générés en JSON

### TP3 - MV
- [ ] PCA explique variance
- [ ] Loadings cohérents
- [ ] Clustering converge
- [ ] Silhouette score calculé
- [ ] Rapport interprétatif généré

### TP4 - ML
- [ ] LogReg s'entraîne
- [ ] Random Forest s'entraîne
- [ ] Métriques cohérentes (0-1)
- [ ] Prédictions fonctionnent
- [ ] Modèle sauvegardé et rechargeable

### TP5 - ML2
- [ ] Grid Search fonctionne
- [ ] Random Search fonctionne
- [ ] Importance native calculée
- [ ] Permutation importance fonctionne
- [ ] Explications locales générées

## Tests de Performance

### Temps de Réponse Attendus

- Génération dataset (1000 lignes) : < 2s
- Clean fit : < 3s
- Clean transform : < 2s
- EDA summary : < 1s
- PCA (1000 lignes, 8 vars) : < 2s
- K-Means (1000 lignes, k=3) : < 2s
- ML train (LogReg) : < 3s
- ML train (RF, 100 arbres) : < 10s
- Tune (Grid Search) : < 60s
- Predict (1 instance) : < 1s

### Limites Connues

- Datasets > 10000 lignes : peut être lent
- Grid Search exhaustif : peut prendre plusieurs minutes
- Graphiques complexes : génération peut être lente

## Debugging

### Logs Utiles

```bash
# Voir tous les logs
docker-compose logs -f api

# Logs uniquement des erreurs
docker-compose logs api | grep ERROR

# Logs de démarrage
docker-compose logs api | head -50
```

### Erreurs Communes

**500 Internal Server Error**
- Vérifier les logs : `docker-compose logs api`
- Souvent : dataset_id invalide ou model_id inexistant

**422 Validation Error**
- Vérifier le schéma JSON
- Lire le message d'erreur détaillé dans la réponse

**404 Not Found**
- Dataset ou model inexistant
- Vérifier l'ID utilisé

## Tests Automatisés (Optionnel)

Pour lancer des tests automatisés, utilisez le script Python :

```bash
python exemples_python.py
```

## Rapport de Test

Après avoir testé, créez un rapport avec :
- Fonctionnalités testées ✓
- Fonctionnalités en échec ✗
- Observations
- Suggestions d'amélioration
