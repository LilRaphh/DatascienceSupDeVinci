# 🌐 Interface Web Interactive

Interface moderne et interactive pour tester tous les TPs de l'API FastAPI Data Scientist.

## ✨ Fonctionnalités

### Design Moderne
- 🎨 Interface colorée et attrayante
- 🌊 Animations fluides et transitions
- 📱 Responsive (fonctionne sur mobile)
- 🎭 Effets visuels (hover, gradients, shadows)

### Fonctionnalités Interactives
- ✅ Navigation intuitive entre les 5 TPs
- ✅ Formulaires dynamiques pour chaque TP
- ✅ Affichage en temps réel des résultats
- ✅ Graphiques interactifs Plotly
- ✅ Toast notifications
- ✅ Indicateur de status API
- ✅ Loading states avec spinner

### Workflow Visuel
- 📊 Étapes de progression pour chaque TP
- 🎯 Indicateurs visuels (complété/actif/en attente)
- 🔄 Gestion automatique du state

## 🚀 Utilisation

### 1. Lancer l'API

```bash
cd ../
docker-compose up -d
```

### 2. Ouvrir l'Interface

Ouvrez simplement le fichier `index.html` dans votre navigateur :

```bash
# Option 1 : Double-clic sur index.html

# Option 2 : Depuis le terminal
open index.html  # macOS
xdg-open index.html  # Linux
start index.html  # Windows

# Option 3 : Serveur HTTP simple
python -m http.server 8080
# Puis ouvrir http://localhost:8080
```

### 3. Utiliser l'Interface

1. **Vérifiez le status** de l'API (indicateur en haut à droite)
2. **Choisissez un TP** en cliquant sur une carte
3. **Suivez le workflow** étape par étape
4. **Visualisez les résultats** en temps réel

## 📁 Structure

```
web-interface/
├── index.html           # Page principale
├── css/
│   └── style.css       # Styles modernes
├── js/
│   ├── config.js       # Configuration et constantes
│   ├── api.js          # Module API (requêtes)
│   ├── ui.js           # Module UI (interface)
│   └── app.js          # Application principale
└── README.md           # Ce fichier
```

## 🎨 Design

### Palette de Couleurs
- **Primary** : #6366f1 (Indigo)
- **Secondary** : #8b5cf6 (Violet)
- **Success** : #10b981 (Green)
- **Warning** : #f59e0b (Amber)
- **Danger** : #ef4444 (Red)

### Typographie
- **Famille** : Inter, SF Pro, Segoe UI
- **Titres** : 800 (Extra Bold)
- **Corps** : 400 (Regular)

### Effets
- **Transitions** : 0.3s cubic-bezier
- **Shadows** : Multi-niveaux
- **Gradients** : Linéaires et radiaux
- **Backdrop blur** : Pour effets vitrés

## 🔧 Workflows par TP

### TP1 - Clean
1. Générer Dataset → Voir défauts
2. Rapport Qualité → Analyser problèmes
3. Créer Pipeline → Définir stratégies
4. Appliquer → Voir résultats

### TP2 - EDA
1. Générer Dataset → Données EDA
2. Statistiques → Résumés numériques
3. Corrélations → Matrice + top paires
4. Graphiques → Plotly interactifs

### TP3 - MV
1. Générer Dataset → 8 variables
2. PCA → Variance + loadings + plot 2D
3. Clustering → K-Means + silhouette
4. Rapport → Interprétation

### TP4 - ML
1. Générer Dataset → Avec target
2. Entraîner → Random Forest
3. Métriques → Accuracy, F1, etc.
4. Prédire → Nouvelles instances

### TP5 - ML2
1. Générer Dataset → Classification
2. Grid Search → Optimisation (1-2 min)
3. Feature Importance → Top features + chart
4. Explicabilité → Contributions locales

## 🎯 Fonctionnalités Détaillées

### Gestion du State
```javascript
STATE = {
    currentTP: 'clean',
    datasetId: 'clean_42_1000_xyz',
    modelId: 'rf_model123',
    cleanerId: 'cleaner_abc',
    apiOnline: true
}
```

### Affichage des Résultats
- **JSON formaté** avec syntax highlighting
- **Métriques** en cards visuelles
- **Graphiques** Plotly zoomables
- **Horodatage** automatique

### Notifications
- **Success** : Vert avec icône check
- **Error** : Rouge avec icône x
- **Warning** : Orange avec icône !
- **Info** : Bleu avec icône i

## 📊 Visualisations

### Types de Graphiques
- **Scatter plots** : PCA projections
- **Bar charts** : Feature importance
- **Histograms** : Distributions
- **Heatmaps** : Corrélations (via EDA)

### Interactivité Plotly
- Zoom et pan
- Hover tooltips
- Légendes cliquables
- Export PNG/SVG

## 🔒 Sécurité

- Pas de données sensibles stockées
- Toutes les requêtes via HTTPS (si configuré)
- Validation côté client
- Gestion d'erreurs robuste

## 🐛 Dépannage

### L'API n'est pas détectée

```bash
# Vérifier que l'API tourne
docker-compose ps

# Vérifier les logs
docker-compose logs api

# Vérifier l'URL dans config.js
# Par défaut : http://localhost:8000
```

### Les graphiques ne s'affichent pas

- Vérifiez la connexion internet (Plotly CDN)
- Ouvrez la console (F12) pour voir les erreurs
- Vérifiez que les données sont bien retournées

### Erreurs CORS

Si vous utilisez un serveur local :
```bash
python -m http.server 8080
```

## 🚀 Améliorations Futures

- [ ] Mode sombre/clair
- [ ] Sauvegarde de session (localStorage)
- [ ] Export des résultats en PDF
- [ ] Historique des actions
- [ ] Comparaison de modèles
- [ ] Authentification utilisateur
- [ ] Websockets pour updates temps réel

## 💡 Conseils d'Utilisation

1. **Commencez par TP1** pour comprendre le workflow
2. **Gardez l'onglet ouvert** pour conserver le state
3. **Utilisez un grand écran** pour mieux visualiser
4. **Explorez les graphiques** (zoom, pan)
5. **Lisez les rapports JSON** pour les détails

## 🎓 Valeur Pédagogique

Cette interface permet de :
- **Comprendre visuellement** chaque étape
- **Tester rapidement** différents paramètres
- **Voir les résultats** immédiatement
- **Apprendre** l'interaction avec une API REST
- **Découvrir** les visualisations de données

## 🤝 Contribution

Pour améliorer l'interface :
1. Modifiez les fichiers dans `css/` et `js/`
2. Testez dans plusieurs navigateurs
3. Vérifiez la responsive design
4. Documentez les changements

## 📝 Licence

Même licence que le projet principal (MIT)

---

**Enjoy l'interface ! 🎉**

Pour plus d'infos sur l'API : voir `../README.md`
