// Configuration globale
const CONFIG = {
    API_URL: 'http://localhost:8000',
    DEFAULT_SEED: 42,
    DEFAULT_N_ROWS: 1000
};

// État global de l'application
const STATE = {
    currentTP: null,
    datasetId: null,
    modelId: null,
    cleanerId: null,
    apiOnline: false
};

// Définition des TPs et leurs workflows
const TP_CONFIGS = {
    clean: {
        title: 'TP1 - Nettoyage de Données',
        description: 'Nettoyage et préparation des données avec gestion des NA, doublons et outliers',
        icon: 'fa-broom',
        steps: [
            { id: 'generate', label: 'Générer Dataset', completed: false },
            { id: 'report', label: 'Rapport Qualité', completed: false },
            { id: 'fit', label: 'Créer Pipeline', completed: false },
            { id: 'transform', label: 'Appliquer Nettoyage', completed: false }
        ]
    },
    eda: {
        title: 'TP2 - Analyse Exploratoire',
        description: 'Statistiques descriptives, corrélations et visualisations interactives',
        icon: 'fa-chart-bar',
        steps: [
            { id: 'generate', label: 'Générer Dataset', completed: false },
            { id: 'summary', label: 'Statistiques', completed: false },
            { id: 'correlation', label: 'Corrélations', completed: false },
            { id: 'plots', label: 'Graphiques', completed: false }
        ]
    },
    mv: {
        title: 'TP3 - Analyse Multivariée',
        description: 'Réduction de dimensionnalité (PCA) et clustering (K-Means)',
        icon: 'fa-project-diagram',
        steps: [
            { id: 'generate', label: 'Générer Dataset', completed: false },
            { id: 'pca', label: 'PCA', completed: false },
            { id: 'cluster', label: 'Clustering', completed: false },
            { id: 'report', label: 'Rapport', completed: false }
        ]
    },
    ml: {
        title: 'TP4 - Machine Learning Baseline',
        description: 'Entraînement de modèles LogReg et Random Forest avec métriques',
        icon: 'fa-brain',
        steps: [
            { id: 'generate', label: 'Générer Dataset', completed: false },
            { id: 'train', label: 'Entraîner Modèle', completed: false },
            { id: 'metrics', label: 'Métriques', completed: false },
            { id: 'predict', label: 'Prédictions', completed: false }
        ]
    },
    ml2: {
        title: 'TP5 - ML Avancé',
        description: 'Optimisation d\'hyperparamètres et explicabilité des modèles',
        icon: 'fa-magic',
        steps: [
            { id: 'generate', label: 'Générer Dataset', completed: false },
            { id: 'tune', label: 'Grid Search', completed: false },
            { id: 'importance', label: 'Feature Importance', completed: false },
            { id: 'explain', label: 'Explicabilité', completed: false }
        ]
    }
};
