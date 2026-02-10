# Guide de Démarrage Rapide

## Mise en route en 3 étapes

### 1. Lancer l'application

```bash
cd fastapi-data-scientist
docker-compose up -d
```

Attendez quelques secondes le temps que le conteneur démarre.

### 2. Vérifier que tout fonctionne

Ouvrez votre navigateur et accédez à :
```
http://localhost:8000/docs
```

Vous devriez voir l'interface Swagger UI avec tous les endpoints.

### 3. Premier test - Générer un dataset

Dans Swagger UI :
1. Cliquez sur `POST /dataset/generate`
2. Cliquez sur "Try it out"
3. Utilisez ce JSON :
```json
{
  "phase": "clean",
  "seed": 42,
  "n": 1000
}
```
4. Cliquez sur "Execute"
5. Copiez le `dataset_id` de la réponse

## Exemples Rapides par TP

### TP1 - Nettoyage

1. Générez un dataset (phase="clean")
2. Testez `/clean/report/{dataset_id}` pour voir les problèmes
3. Testez `/clean/fit` avec votre dataset_id
4. Copiez le `cleaner_id` retourné
5. Testez `/clean/transform` avec le cleaner_id

### TP2 - EDA

1. Générez un dataset (phase="eda")
2. Testez `/eda/summary`
3. Testez `/eda/groupby` avec by="segment"
4. Testez `/eda/correlation`
5. Testez `/eda/plots` pour générer des graphiques

### TP3 - Analyse Multivariée

1. Générez un dataset (phase="mv")
2. Testez `/mv/pca/fit_transform` avec n_components=2
3. Testez `/mv/cluster/kmeans` avec k=3
4. Consultez `/mv/report/{dataset_id}`

### TP4 - ML Baseline

1. Générez un dataset (phase="ml")
2. Testez `/ml/train` avec model_type="rf"
3. Copiez le `model_id` retourné
4. Testez `/ml/metrics/{model_id}`
5. Testez `/ml/predict` avec des données

### TP5 - ML Avancé

1. Utilisez le dataset du TP4
2. Testez `/ml2/tune` avec search="grid"
3. Copiez le `best_model_id`
4. Testez `/ml2/feature-importance/{model_id}`
5. Testez `/ml2/explain-instance` pour une explication

## Arrêter l'application

```bash
docker-compose down
```

## Logs et Debug

Voir les logs :
```bash
docker-compose logs -f api
```

## Commandes Utiles

```bash
# Rebuild complet
docker-compose up --build -d

# Restart
docker-compose restart

# Voir les conteneurs actifs
docker-compose ps

# Accéder au conteneur
docker-compose exec api bash
```

## Structure JSON des Requêtes

Toutes les requêtes suivent le même format :
- **dataset_id** : En query parameter (`?dataset_id=xxx`)
- **params** : Dans le body JSON
- **data** : Dans le body JSON pour les prédictions

## Endpoints les Plus Utiles

- `/docs` - Documentation interactive complète
- `/` - Informations sur l'API
- `/health` - Vérification que l'API fonctionne

## Astuces

1. **Reproductibilité** : Utilisez toujours le même seed pour obtenir les mêmes datasets
2. **Dataset IDs** : Gardez une trace de vos dataset_ids dans un fichier texte
3. **Model IDs** : Notez les model_ids de vos meilleurs modèles
4. **Graphiques** : Les graphiques sont en format Plotly JSON dans `artifacts`

## Problèmes Courants

**Port 8000 déjà utilisé** :
```bash
# Modifier le port dans docker-compose.yml
ports:
  - "8001:8000"  # Utiliser 8001 au lieu de 8000
```

**Conteneur ne démarre pas** :
```bash
docker-compose logs api
# Vérifier les erreurs
```

**Erreur 404 dataset not found** :
- Vérifiez que vous utilisez le bon dataset_id
- Générez un nouveau dataset si nécessaire

## Pour Aller Plus Loin

Consultez le README.md complet pour :
- Exemples détaillés avec curl
- Toute la documentation des endpoints
- Architecture du code
- Guide de contribution

Bon coding !
