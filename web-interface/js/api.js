// Module API pour gérer toutes les requêtes
const API = {
    // Vérification de l'état de l'API
    async checkHealth() {
        try {
            const response = await fetch(`${CONFIG.API_URL}/health`);
            return response.ok;
        } catch (error) {
            console.error('API non disponible:', error);
            return false;
        }
    },

    // Génération de dataset
    async generateDataset(phase, seed = CONFIG.DEFAULT_SEED, n = CONFIG.DEFAULT_N_ROWS) {
        const response = await fetch(`${CONFIG.API_URL}/dataset/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phase, seed, n })
        });
        return await response.json();
    },

    // TP1 - Clean
    async getCleanReport(datasetId) {
        const response = await fetch(`${CONFIG.API_URL}/clean/report/${datasetId}`);
        return await response.json();
    },

    async cleanFit(datasetId, params) {
        const response = await fetch(`${CONFIG.API_URL}/clean/fit?dataset_id=${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return await response.json();
    },

    async cleanTransform(datasetId, cleanerId) {
        const response = await fetch(`${CONFIG.API_URL}/clean/transform?dataset_id=${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cleaner_id: cleanerId })
        });
        return await response.json();
    },

    // TP2 - EDA
    async edaSummary(datasetId) {
        const response = await fetch(`${CONFIG.API_URL}/eda/summary?dataset_id=${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        return await response.json();
    },

    async edaCorrelation(datasetId) {
        const response = await fetch(`${CONFIG.API_URL}/eda/correlation?dataset_id=${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ method: 'pearson', threshold: 0.3 })
        });
        return await response.json();
    },

    async edaPlots(datasetId, plotType, params) {
        const response = await fetch(`${CONFIG.API_URL}/eda/plots?dataset_id=${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plot_types: [plotType], ...params })
        });
        return await response.json();
    },

    // TP3 - MV
    async mvPca(datasetId, nComponents = 2) {
        const response = await fetch(`${CONFIG.API_URL}/mv/pca/fit_transform?dataset_id=${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_components: nComponents, scale: true })
        });
        return await response.json();
    },

    async mvCluster(datasetId, k = 3) {
        const response = await fetch(`${CONFIG.API_URL}/mv/cluster/kmeans?dataset_id=${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ k, scale: true })
        });
        return await response.json();
    },

    // TP4 - ML
    async mlTrain(datasetId, modelType = 'rf') {
        const response = await fetch(`${CONFIG.API_URL}/ml/train?dataset_id=${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                model_type: modelType,
                test_size: 0.2,
                rf_n_estimators: 100
            })
        });
        return await response.json();
    },

    async mlMetrics(modelId) {
        const response = await fetch(`${CONFIG.API_URL}/ml/metrics/${modelId}`);
        return await response.json();
    },

    async mlPredict(modelId, data) {
        const response = await fetch(`${CONFIG.API_URL}/ml/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId, data })
        });
        return await response.json();
    },

    // TP5 - ML2
    async ml2Tune(datasetId) {
        const response = await fetch(`${CONFIG.API_URL}/ml2/tune?dataset_id=${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                model_type: 'rf',
                search: 'grid',
                cv: 3,
                scoring: 'f1'
            })
        });
        return await response.json();
    },

    async ml2FeatureImportance(modelId) {
        const response = await fetch(`${CONFIG.API_URL}/ml2/feature-importance/${modelId}`);
        return await response.json();
    },

    async ml2Explain(modelId, instance) {
        const response = await fetch(`${CONFIG.API_URL}/ml2/explain-instance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId, instance })
        });
        return await response.json();
    }
};
