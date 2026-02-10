// Application principale
document.addEventListener('DOMContentLoaded', async () => {
    // Vérifier l'état de l'API
    const isOnline = await API.checkHealth();
    STATE.apiOnline = isOnline;
    UI.updateApiStatus(isOnline);
    
    if (!isOnline) {
        UI.showToast('API non disponible. Lancez docker-compose up -d', 'error');
    }
});

// Charger un TP
function loadTP(tp) {
    STATE.currentTP = tp;
    const config = TP_CONFIGS[tp];
    
    // Mettre à jour l'interface
    document.getElementById('workspaceTitle').textContent = config.title;
    document.getElementById('workspaceDescription').textContent = config.description;
    
    // Générer le formulaire
    UI.generateConfigForm(tp);
    
    // Mettre à jour les étapes
    UI.updateWorkflowSteps(tp, []);
    
    // Afficher le workspace
    UI.showSection('workspace');
    
    // Réinitialiser les résultats
    UI.clearResults();
    
    // Masquer la zone de visualisation
    document.getElementById('vizArea').style.display = 'none';
}

// Retour à l'accueil
function backToHome() {
    document.querySelectorAll('section').forEach(section => {
        section.style.display = section.classList.contains('hero') || 
                                section.classList.contains('tp-navigation') ? 'block' : 'none';
    });
}

// ===== Handlers pour chaque TP =====

// Génération de dataset (commun à tous les TPs)
async function handleGenerateDataset() {
    if (!STATE.apiOnline) {
        UI.showToast('API non disponible', 'error');
        return;
    }
    
    UI.showLoading(true);
    try {
        const result = await API.generateDataset(STATE.currentTP);
        STATE.datasetId = result.result.dataset_id;
        
        UI.showToast('Dataset généré avec succès!', 'success');
        UI.displayResults('Dataset généré', {
            dataset_id: result.result.dataset_id,
            n_rows: result.result.n_rows,
            n_cols: result.result.n_cols,
            columns: result.result.columns
        });
        
        UI.updateWorkflowSteps(STATE.currentTP, ['generate']);
        
        // Afficher les options spécifiques au TP
        const optionsId = STATE.currentTP + 'Options';
        const buttonsId = STATE.currentTP + 'Buttons';
        const options = document.getElementById(optionsId);
        const buttons = document.getElementById(buttonsId);
        if (options) options.style.display = 'block';
        if (buttons) buttons.style.display = 'block';
        
    } catch (error) {
        UI.showToast('Erreur lors de la génération', 'error');
        console.error(error);
    }
    UI.showLoading(false);
}

// ===== TP1 - Clean =====
async function handleCleanReport() {
    UI.showLoading(true);
    try {
        const result = await API.getCleanReport(STATE.datasetId);
        UI.displayResults('Rapport de qualité', result.result);
        UI.updateWorkflowSteps('clean', ['generate', 'report']);
    } catch (error) {
        UI.showToast('Erreur lors du rapport', 'error');
    }
    UI.showLoading(false);
}

async function handleCleanFit() {
    UI.showLoading(true);
    try {
        const strategy = document.getElementById('imputeStrategy').value;
        const result = await API.cleanFit(STATE.datasetId, {
            impute_strategy: strategy,
            outlier_strategy: 'clip',
            categorical_strategy: 'one_hot'
        });
        
        STATE.cleanerId = result.result.cleaner_id;
        UI.showToast('Pipeline créé!', 'success');
        UI.displayResults('Pipeline de nettoyage', {
            cleaner_id: result.result.cleaner_id,
            rules: result.result.rules
        });
        UI.updateWorkflowSteps('clean', ['generate', 'report', 'fit']);
    } catch (error) {
        UI.showToast('Erreur lors de la création du pipeline', 'error');
    }
    UI.showLoading(false);
}

// ===== TP2 - EDA =====
async function handleEdaSummary() {
    UI.showLoading(true);
    try {
        const result = await API.edaSummary(STATE.datasetId);
        UI.displayResults('Statistiques descriptives', result.result.summaries);
        UI.updateWorkflowSteps('eda', ['generate', 'summary']);
    } catch (error) {
        UI.showToast('Erreur lors du calcul des statistiques', 'error');
    }
    UI.showLoading(false);
}

async function handleEdaCorrelation() {
    UI.showLoading(true);
    try {
        const result = await API.edaCorrelation(STATE.datasetId);
        UI.displayResults('Corrélations', {
            top_correlations: result.result.top_correlations
        });
        UI.updateWorkflowSteps('eda', ['generate', 'summary', 'correlation']);
    } catch (error) {
        UI.showToast('Erreur lors du calcul des corrélations', 'error');
    }
    UI.showLoading(false);
}

async function handleEdaPlots() {
    UI.showLoading(true);
    try {
        const result = await API.edaPlots(STATE.datasetId, 'histogram', {
            numeric_var: 'income'
        });
        
        if (result.artifacts && result.artifacts.histogram) {
            UI.displayPlot(result.artifacts.histogram, 'Distribution du revenu');
            UI.showToast('Graphique généré!', 'success');
        }
        UI.updateWorkflowSteps('eda', ['generate', 'summary', 'correlation', 'plots']);
    } catch (error) {
        UI.showToast('Erreur lors de la génération des graphiques', 'error');
    }
    UI.showLoading(false);
}

// ===== TP3 - MV =====
async function handleMvPca() {
    UI.showLoading(true);
    try {
        const result = await API.mvPca(STATE.datasetId, 2);
        UI.displayResults('PCA', {
            variance_explained: result.result.total_variance_explained,
            components: result.result.components_info
        });
        
        // Créer un scatter plot de PC1 vs PC2
        const projection = result.result.projection;
        const trace = {
            x: projection.map(p => p.PC1),
            y: projection.map(p => p.PC2),
            mode: 'markers',
            type: 'scatter',
            marker: { size: 5, color: '#6366f1' }
        };
        
        const layout = {
            title: 'Projection PCA (PC1 vs PC2)',
            xaxis: { title: 'PC1' },
            yaxis: { title: 'PC2' }
        };
        
        Plotly.newPlot('plotlyChart', [trace], layout);
        document.getElementById('vizArea').style.display = 'block';
        
        UI.updateWorkflowSteps('mv', ['generate', 'pca']);
    } catch (error) {
        UI.showToast('Erreur lors de la PCA', 'error');
    }
    UI.showLoading(false);
}

async function handleMvCluster() {
    UI.showLoading(true);
    try {
        const result = await API.mvCluster(STATE.datasetId, 3);
        UI.displayMetrics({
            'Silhouette Score': result.result.silhouette_score,
            'Inertie': result.result.inertia
        });
        UI.displayResults('Clustering', {
            clusters_info: result.result.clusters_info
        });
        UI.updateWorkflowSteps('mv', ['generate', 'pca', 'cluster']);
    } catch (error) {
        UI.showToast('Erreur lors du clustering', 'error');
    }
    UI.showLoading(false);
}

// ===== TP4 - ML =====
async function handleMlTrain() {
    UI.showLoading(true);
    try {
        const result = await API.mlTrain(STATE.datasetId, 'rf');
        STATE.modelId = result.result.model_id;
        
        UI.displayMetrics({
            'Accuracy': result.result.metrics_test.accuracy,
            'Precision': result.result.metrics_test.precision,
            'Recall': result.result.metrics_test.recall,
            'F1-Score': result.result.metrics_test.f1_score
        });
        
        UI.displayResults('Modèle entraîné', {
            model_id: result.result.model_id,
            training_time: result.result.training_time
        });
        
        document.getElementById('btnPredict').disabled = false;
        UI.updateWorkflowSteps('ml', ['generate', 'train']);
    } catch (error) {
        UI.showToast('Erreur lors de l\'entraînement', 'error');
    }
    UI.showLoading(false);
}

async function handleMlPredict() {
    UI.showLoading(true);
    try {
        const testData = [{
            x1: 15, x2: 22, x3: 10, x4: 5, x5: 30, x6: 12, segment: 'A'
        }];
        
        const result = await API.mlPredict(STATE.modelId, testData);
        UI.displayResults('Prédictions', result.result.predictions);
        UI.updateWorkflowSteps('ml', ['generate', 'train', 'metrics', 'predict']);
    } catch (error) {
        UI.showToast('Erreur lors de la prédiction', 'error');
    }
    UI.showLoading(false);
}

// ===== TP5 - ML2 =====
async function handleMl2Tune() {
    UI.showLoading(true);
    UI.showToast('Grid Search en cours (peut prendre 1-2 minutes)...', 'warning');
    try {
        const result = await API.ml2Tune(STATE.datasetId);
        STATE.modelId = result.result.best_model_id;
        
        UI.displayMetrics({
            'Best Score (F1)': result.result.best_score,
            'Total Fits': result.result.total_fits,
            'Tuning Time': result.result.tuning_time
        });
        
        UI.displayResults('Grid Search', {
            best_model_id: result.result.best_model_id,
            best_params: result.result.best_params
        });
        
        document.getElementById('btnImportance').disabled = false;
        UI.updateWorkflowSteps('ml2', ['generate', 'tune']);
        UI.showToast('Optimisation terminée!', 'success');
    } catch (error) {
        UI.showToast('Erreur lors de l\'optimisation', 'error');
    }
    UI.showLoading(false);
}

async function handleMl2Importance() {
    UI.showLoading(true);
    try {
        const result = await API.ml2FeatureImportance(STATE.modelId);
        
        // Créer un bar chart des importances
        const importances = result.result.importances.slice(0, 10);
        const trace = {
            x: importances.map(i => i.importance),
            y: importances.map(i => i.feature),
            type: 'bar',
            orientation: 'h',
            marker: { color: '#6366f1' }
        };
        
        const layout = {
            title: 'Top 10 Feature Importances',
            xaxis: { title: 'Importance' },
            yaxis: { autorange: 'reversed' }
        };
        
        Plotly.newPlot('plotlyChart', [trace], layout);
        document.getElementById('vizArea').style.display = 'block';
        
        UI.displayResults('Feature Importance', {
            top_features: result.result.top_features
        });
        
        UI.updateWorkflowSteps('ml2', ['generate', 'tune', 'importance']);
    } catch (error) {
        UI.showToast('Erreur lors du calcul d\'importance', 'error');
    }
    UI.showLoading(false);
}
