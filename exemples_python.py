"""
Exemples d'utilisation de l'API FastAPI Data Scientist avec Python.
Assurez-vous que l'API est lancee avant d'executer ces exemples.
"""

import requests
import json

# URL de base de l'API
BASE_URL = "http://localhost:8000"


def exemple_tp1_nettoyage():
    """
    Exemple complet du TP1 - Nettoyage de donnees.
    """
    print("=== TP1 - NETTOYAGE DE DONNEES ===\n")
    
    # 1. Generer un dataset
    print("1. Generation du dataset...")
    response = requests.post(
        f"{BASE_URL}/dataset/generate",
        json={
            "phase": "clean",
            "seed": 42,
            "n": 1000
        }
    )
    dataset_id = response.json()["result"]["dataset_id"]
    print(f"Dataset genere: {dataset_id}\n")
    
    # 2. Rapport de qualite
    print("2. Rapport de qualite...")
    response = requests.get(f"{BASE_URL}/clean/report/{dataset_id}")
    report = response.json()["result"]
    print(f"Doublons: {report['n_duplicates']}")
    print(f"Valeurs manquantes totales: {sum(report['missing_values'].values())}\n")
    
    # 3. Apprendre un pipeline de nettoyage
    print("3. Apprentissage du pipeline...")
    response = requests.post(
        f"{BASE_URL}/clean/fit",
        params={"dataset_id": dataset_id},
        json={
            "impute_strategy": "mean",
            "outlier_strategy": "clip",
            "categorical_strategy": "one_hot",
            "remove_duplicates": True
        }
    )
    cleaner_id = response.json()["result"]["cleaner_id"]
    print(f"Cleaner cree: {cleaner_id}\n")
    
    # 4. Appliquer le nettoyage
    print("4. Application du nettoyage...")
    response = requests.post(
        f"{BASE_URL}/clean/transform",
        params={"dataset_id": dataset_id},
        json={"cleaner_id": cleaner_id}
    )
    result = response.json()["result"]
    print(f"Lignes avant: {result['n_rows_before']}")
    print(f"Lignes apres: {result['n_rows_after']}")
    print(f"Transformations: {result['counters']}\n")


def exemple_tp2_eda():
    """
    Exemple complet du TP2 - EDA.
    """
    print("=== TP2 - ANALYSE EXPLORATOIRE ===\n")
    
    # 1. Generer un dataset
    print("1. Generation du dataset...")
    response = requests.post(
        f"{BASE_URL}/dataset/generate",
        json={"phase": "eda", "seed": 42, "n": 1000}
    )
    dataset_id = response.json()["result"]["dataset_id"]
    print(f"Dataset genere: {dataset_id}\n")
    
    # 2. Statistiques descriptives
    print("2. Statistiques descriptives...")
    response = requests.post(
        f"{BASE_URL}/eda/summary",
        params={"dataset_id": dataset_id}
    )
    summary = response.json()["result"]["summaries"]
    print(f"Variables analysees: {len(summary)}")
    print(f"Exemple - age: mean={summary['age']['mean']:.2f}, std={summary['age']['std']:.2f}\n")
    
    # 3. Groupby
    print("3. Agregation par segment...")
    response = requests.post(
        f"{BASE_URL}/eda/groupby",
        params={"dataset_id": dataset_id},
        json={
            "by": "segment",
            "metrics": ["mean", "median"]
        }
    )
    groupby = response.json()["result"]["aggregations"]
    print(f"Nombre de groupes: {len(groupby)}")
    for group in groupby:
        print(f"  Segment {group['segment']}: income_mean={group['income_mean']:.2f}")
    print()
    
    # 4. Correlation
    print("4. Matrice de correlation...")
    response = requests.post(
        f"{BASE_URL}/eda/correlation",
        params={"dataset_id": dataset_id},
        json={"method": "pearson", "threshold": 0.3}
    )
    corr = response.json()["result"]
    print(f"Top correlations trouvees: {len(corr['top_correlations'])}\n")


def exemple_tp3_multivarié():
    """
    Exemple complet du TP3 - Analyse multivariee.
    """
    print("=== TP3 - ANALYSE MULTIVARIEE ===\n")
    
    # 1. Generer un dataset
    print("1. Generation du dataset...")
    response = requests.post(
        f"{BASE_URL}/dataset/generate",
        json={"phase": "mv", "seed": 42, "n": 1000}
    )
    dataset_id = response.json()["result"]["dataset_id"]
    print(f"Dataset genere: {dataset_id}\n")
    
    # 2. PCA
    print("2. PCA...")
    response = requests.post(
        f"{BASE_URL}/mv/pca/fit_transform",
        params={"dataset_id": dataset_id},
        json={"n_components": 3, "scale": True}
    )
    pca = response.json()["result"]
    print(f"Variance expliquee: {pca['total_variance_explained']:.2f}%")
    for comp in pca['components_info']:
        print(f"  PC{comp['component']}: {comp['explained_variance_ratio']:.2f}%")
        top_vars = [l['variable'] for l in comp['top_loadings'][:3]]
        print(f"    Top variables: {', '.join(top_vars)}")
    print()
    
    # 3. Clustering
    print("3. Clustering K-Means...")
    response = requests.post(
        f"{BASE_URL}/mv/cluster/kmeans",
        params={"dataset_id": dataset_id},
        json={"k": 3, "scale": True}
    )
    clustering = response.json()["result"]
    print(f"Silhouette score: {clustering['silhouette_score']:.3f}")
    for cluster in clustering['clusters_info']:
        print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} points ({cluster['percentage']:.1f}%)")
    print()


def exemple_tp4_ml_baseline():
    """
    Exemple complet du TP4 - ML Baseline.
    """
    print("=== TP4 - ML BASELINE ===\n")
    
    # 1. Generer un dataset
    print("1. Generation du dataset...")
    response = requests.post(
        f"{BASE_URL}/dataset/generate",
        json={"phase": "ml", "seed": 42, "n": 1000}
    )
    dataset_id = response.json()["result"]["dataset_id"]
    print(f"Dataset genere: {dataset_id}\n")
    
    # 2. Entrainer un Random Forest
    print("2. Entrainement du modele Random Forest...")
    response = requests.post(
        f"{BASE_URL}/ml/train",
        params={"dataset_id": dataset_id},
        json={
            "model_type": "rf",
            "test_size": 0.2,
            "rf_n_estimators": 100,
            "rf_max_depth": 10
        }
    )
    result = response.json()["result"]
    model_id = result["model_id"]
    print(f"Modele entraine: {model_id}")
    print(f"Accuracy test: {result['metrics_test']['accuracy']:.3f}")
    print(f"F1-score test: {result['metrics_test']['f1_score']:.3f}\n")
    
    # 3. Faire des predictions
    print("3. Predictions...")
    response = requests.post(
        f"{BASE_URL}/ml/predict",
        json={
            "model_id": model_id,
            "data": [
                {"x1": 15.0, "x2": 22.0, "x3": 10.0, "x4": 5.0, "x5": 30.0, "x6": 12.0, "segment": "A"},
                {"x1": 8.0, "x2": 18.0, "x3": 15.0, "x4": 3.0, "x5": 25.0, "x6": 8.0, "segment": "B"}
            ]
        }
    )
    predictions = response.json()["result"]["predictions"]
    for i, pred in enumerate(predictions):
        print(f"  Instance {i+1}: classe={pred['prediction']}, proba={pred['probability']}")
    print()


def exemple_tp5_ml_avance():
    """
    Exemple complet du TP5 - ML Avance.
    """
    print("=== TP5 - ML AVANCE ===\n")
    
    # 1. Generer un dataset
    print("1. Generation du dataset...")
    response = requests.post(
        f"{BASE_URL}/dataset/generate",
        json={"phase": "ml2", "seed": 42, "n": 1000}
    )
    dataset_id = response.json()["result"]["dataset_id"]
    print(f"Dataset genere: {dataset_id}\n")
    
    # 2. Optimisation des hyperparametres
    print("2. Optimisation avec Grid Search...")
    response = requests.post(
        f"{BASE_URL}/ml2/tune",
        params={"dataset_id": dataset_id},
        json={
            "model_type": "rf",
            "search": "grid",
            "cv": 3,
            "scoring": "f1"
        }
    )
    tune_result = response.json()["result"]
    best_model_id = tune_result["best_model_id"]
    print(f"Meilleur modele: {best_model_id}")
    print(f"Meilleur score CV: {tune_result['best_score']:.3f}")
    print(f"Meilleurs params: {tune_result['best_params']}\n")
    
    # 3. Importance des features
    print("3. Importance des features...")
    response = requests.get(f"{BASE_URL}/ml2/feature-importance/{best_model_id}")
    importance = response.json()["result"]
    print(f"Top 5 features: {', '.join(importance['top_features'])}\n")
    
    # 4. Explication d'une instance
    print("4. Explication d'une prediction...")
    response = requests.post(
        f"{BASE_URL}/ml2/explain-instance",
        json={
            "model_id": best_model_id,
            "instance": {
                "x1": 15.0, "x2": 22.0, "x3": 10.0,
                "x4": 5.0, "x5": 30.0, "x6": 12.0,
                "segment": "A"
            }
        }
    )
    explanation = response.json()["result"]
    print(f"Prediction: {explanation['prediction']}")
    print(f"Probabilite: {explanation['probability']:.3f}")
    print(f"Resume: {explanation['explanation_summary']}\n")


def workflow_complet():
    """
    Exemple d'un workflow complet de A a Z.
    """
    print("=== WORKFLOW COMPLET ===\n")
    
    # 1. Generation et nettoyage
    print("ETAPE 1: Generation et nettoyage des donnees")
    response = requests.post(
        f"{BASE_URL}/dataset/generate",
        json={"phase": "clean", "seed": 42, "n": 1000}
    )
    dataset_id = response.json()["result"]["dataset_id"]
    
    response = requests.post(
        f"{BASE_URL}/clean/fit",
        params={"dataset_id": dataset_id},
        json={"impute_strategy": "mean", "outlier_strategy": "clip"}
    )
    cleaner_id = response.json()["result"]["cleaner_id"]
    print(f"  Dataset: {dataset_id}, Cleaner: {cleaner_id}\n")
    
    # 2. EDA
    print("ETAPE 2: Analyse exploratoire")
    response = requests.post(
        f"{BASE_URL}/dataset/generate",
        json={"phase": "eda", "seed": 42, "n": 1000}
    )
    eda_dataset_id = response.json()["result"]["dataset_id"]
    
    response = requests.post(
        f"{BASE_URL}/eda/summary",
        params={"dataset_id": eda_dataset_id}
    )
    print(f"  Variables analysees: {response.json()['result']['n_cols']}\n")
    
    # 3. ML
    print("ETAPE 3: Machine Learning")
    response = requests.post(
        f"{BASE_URL}/dataset/generate",
        json={"phase": "ml", "seed": 42, "n": 1000}
    )
    ml_dataset_id = response.json()["result"]["dataset_id"]
    
    response = requests.post(
        f"{BASE_URL}/ml2/tune",
        params={"dataset_id": ml_dataset_id},
        json={"model_type": "rf", "search": "grid", "cv": 3}
    )
    best_model_id = response.json()["result"]["best_model_id"]
    print(f"  Modele optimise: {best_model_id}\n")
    
    print("Workflow termine avec succes!")


if __name__ == "__main__":
    print("Exemples d'utilisation de l'API FastAPI Data Scientist\n")
    print("Assurez-vous que l'API est lancee sur http://localhost:8000\n")
    print("=" * 60)
    print()
    
    try:
        # Verifier que l'API est accessible
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("ERREUR: L'API ne repond pas correctement")
            exit(1)
        
        # Executer les exemples
        # Decommentez les exemples que vous voulez executer
        
        exemple_tp1_nettoyage()
        # exemple_tp2_eda()
        # exemple_tp3_multivarié()
        # exemple_tp4_ml_baseline()
        # exemple_tp5_ml_avance()
        # workflow_complet()
        
    except requests.exceptions.ConnectionError:
        print("ERREUR: Impossible de se connecter a l'API")
        print("Assurez-vous que l'API est lancee avec: docker-compose up -d")
