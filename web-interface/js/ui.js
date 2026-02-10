// Module UI pour gérer l'interface
const UI = {
    // Afficher/masquer les sections
    showSection(sectionId) {
        document.querySelectorAll('section').forEach(section => {
            section.style.display = 'none';
        });
        document.getElementById(sectionId).style.display = 'block';
    },

    // Toast notifications
    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast ${type} show`;
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    },

    // Loading overlay
    showLoading(show = true) {
        const overlay = document.getElementById('loadingOverlay');
        overlay.classList.toggle('show', show);
    },

    // Mettre à jour le status de l'API
    updateApiStatus(online) {
        const statusElement = document.getElementById('apiStatus');
        const icon = statusElement.querySelector('i');
        const text = statusElement.querySelector('span');
        
        if (online) {
            statusElement.classList.add('online');
            statusElement.classList.remove('offline');
            text.textContent = 'API En ligne';
        } else {
            statusElement.classList.add('offline');
            statusElement.classList.remove('online');
            text.textContent = 'API Hors ligne';
        }
    },

    // Afficher les résultats
    displayResults(title, data) {
        const resultsPanel = document.getElementById('resultsPanel');
        
        const resultHTML = `
            <div class="result-item">
                <div class="result-header">
                    <div class="result-title">${title}</div>
                    <div class="result-time">${new Date().toLocaleTimeString()}</div>
                </div>
                <div class="result-content">
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                </div>
            </div>
        `;
        
        resultsPanel.innerHTML = resultHTML + resultsPanel.innerHTML;
    },

    // Afficher des métriques
    displayMetrics(metrics) {
        const resultsPanel = document.getElementById('resultsPanel');
        
        let metricsHTML = '<div class="metric-grid">';
        for (const [key, value] of Object.entries(metrics)) {
            const displayValue = typeof value === 'number' ? value.toFixed(3) : value;
            metricsHTML += `
                <div class="metric-card">
                    <div class="metric-label">${key}</div>
                    <div class="metric-value">${displayValue}</div>
                </div>
            `;
        }
        metricsHTML += '</div>';
        
        resultsPanel.innerHTML = metricsHTML + resultsPanel.innerHTML;
    },

    // Afficher un graphique Plotly
    displayPlot(plotData, title = 'Visualisation') {
        const vizArea = document.getElementById('vizArea');
        const plotlyChart = document.getElementById('plotlyChart');
        
        vizArea.style.display = 'block';
        
        if (typeof plotData === 'string') {
            plotData = JSON.parse(plotData);
        }
        
        Plotly.newPlot(plotlyChart, plotData.data, plotData.layout || {
            title: title,
            autosize: true
        });
    },

    // Générer le formulaire pour chaque TP
    generateConfigForm(tp) {
        const configPanel = document.getElementById('configPanel');
        
        const forms = {
            clean: `
                <div class="form-group">
                    <button class="btn" onclick="handleGenerateDataset()">
                        <i class="fas fa-database"></i> Générer Dataset
                    </button>
                </div>
                <div class="form-group" id="cleanOptions" style="display:none;">
                    <label class="form-label">Stratégie d'imputation</label>
                    <select class="form-select" id="imputeStrategy">
                        <option value="mean">Moyenne</option>
                        <option value="median">Médiane</option>
                    </select>
                </div>
                <div class="form-group" id="cleanButtons" style="display:none;">
                    <button class="btn" onclick="handleCleanReport()">
                        <i class="fas fa-file-alt"></i> Voir Rapport
                    </button>
                    <button class="btn btn-success" onclick="handleCleanFit()">
                        <i class="fas fa-tools"></i> Créer Pipeline
                    </button>
                </div>
            `,
            eda: `
                <div class="form-group">
                    <button class="btn" onclick="handleGenerateDataset()">
                        <i class="fas fa-database"></i> Générer Dataset
                    </button>
                </div>
                <div class="form-group" id="edaButtons" style="display:none;">
                    <button class="btn" onclick="handleEdaSummary()">
                        <i class="fas fa-chart-bar"></i> Statistiques
                    </button>
                    <button class="btn" onclick="handleEdaCorrelation()">
                        <i class="fas fa-chart-line"></i> Corrélations
                    </button>
                    <button class="btn" onclick="handleEdaPlots()">
                        <i class="fas fa-chart-pie"></i> Graphiques
                    </button>
                </div>
            `,
            mv: `
                <div class="form-group">
                    <button class="btn" onclick="handleGenerateDataset()">
                        <i class="fas fa-database"></i> Générer Dataset
                    </button>
                </div>
                <div class="form-group" id="mvButtons" style="display:none;">
                    <button class="btn" onclick="handleMvPca()">
                        <i class="fas fa-project-diagram"></i> Lancer PCA
                    </button>
                    <button class="btn" onclick="handleMvCluster()">
                        <i class="fas fa-layer-group"></i> Clustering K-Means
                    </button>
                </div>
            `,
            ml: `
                <div class="form-group">
                    <button class="btn" onclick="handleGenerateDataset()">
                        <i class="fas fa-database"></i> Générer Dataset
                    </button>
                </div>
                <div class="form-group" id="mlButtons" style="display:none;">
                    <button class="btn btn-success" onclick="handleMlTrain()">
                        <i class="fas fa-brain"></i> Entraîner Modèle
                    </button>
                    <button class="btn" id="btnPredict" onclick="handleMlPredict()" disabled>
                        <i class="fas fa-magic"></i> Prédire
                    </button>
                </div>
            `,
            ml2: `
                <div class="form-group">
                    <button class="btn" onclick="handleGenerateDataset()">
                        <i class="fas fa-database"></i> Générer Dataset
                    </button>
                </div>
                <div class="form-group" id="ml2Buttons" style="display:none;">
                    <button class="btn btn-success" onclick="handleMl2Tune()">
                        <i class="fas fa-cogs"></i> Grid Search
                    </button>
                    <button class="btn" id="btnImportance" onclick="handleMl2Importance()" disabled>
                        <i class="fas fa-chart-bar"></i> Feature Importance
                    </button>
                </div>
            `
        };
        
        configPanel.innerHTML = forms[tp] || '<p>Configuration non disponible</p>';
    },

    // Mettre à jour les étapes du workflow
    updateWorkflowSteps(tp, completedSteps = []) {
        const stepsContainer = document.getElementById('workflowSteps');
        const config = TP_CONFIGS[tp];
        
        let stepsHTML = '';
        config.steps.forEach((step, index) => {
            const isCompleted = completedSteps.includes(step.id);
            const isActive = completedSteps.length === index;
            
            stepsHTML += `
                <div class="workflow-step ${isCompleted ? 'completed' : ''} ${isActive ? 'active' : ''}">
                    <div class="step-circle">${index + 1}</div>
                    <div class="step-label">${step.label}</div>
                </div>
            `;
        });
        
        stepsContainer.innerHTML = stepsHTML;
    },

    // Effacer les résultats
    clearResults() {
        const resultsPanel = document.getElementById('resultsPanel');
        resultsPanel.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <p>Aucun résultat</p>
            </div>
        `;
    }
};

// Fonction pour effacer les résultats (globale pour onclick)
function clearResults() {
    UI.clearResults();
}
