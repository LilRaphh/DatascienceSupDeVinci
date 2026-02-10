"""
Service d'Analyse Multivariee (TP3 - phase mv).
Gere PCA et clustering K-Means.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from app.schemas.mv import (
    MvPcaParams,
    MvClusterParams,
    MvPcaResult,
    MvClusterResult,
    MvReportResult,
    PcaComponent,
    PcaLoading,
    ClusterInfo
)


class MvService:
    """Service d'analyse multivariee."""
    
    def pca_fit_transform(
        self,
        df: pd.DataFrame,
        params: MvPcaParams
    ) -> MvPcaResult:
        """
        Effectue une PCA et projette les donnees.
        
        Args:
            df: DataFrame avec variables numeriques
            params: Parametres de la PCA
            
        Returns:
            MvPcaResult avec projection et loadings
        """
        # Selectionner uniquement les colonnes numeriques
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        # Standardiser si demande
        if params.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(numeric_df)
        else:
            X = numeric_df.values
        
        # Appliquer PCA
        pca = PCA(n_components=params.n_components)
        X_pca = pca.fit_transform(X)
        
        # Creer le dataframe de projection
        projection_df = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(params.n_components)]
        )
        projection = projection_df.to_dict('records')
        
        # Calculer les loadings et composantes
        components_info = []
        for i in range(params.n_components):
            # Top loadings pour cette composante
            loadings_values = pca.components_[i]
            top_indices = np.argsort(np.abs(loadings_values))[-5:][::-1]
            
            top_loadings = [
                PcaLoading(
                    variable=numeric_df.columns[idx],
                    loading=float(loadings_values[idx])
                )
                for idx in top_indices
            ]
            
            components_info.append(
                PcaComponent(
                    component=i + 1,
                    explained_variance=float(pca.explained_variance_[i]),
                    explained_variance_ratio=float(pca.explained_variance_ratio_[i] * 100),
                    top_loadings=top_loadings
                )
            )
        
        total_variance = float(sum(pca.explained_variance_ratio_) * 100)
        
        return MvPcaResult(
            n_components=params.n_components,
            total_variance_explained=total_variance,
            components_info=components_info,
            projection=projection
        )
    
    def cluster_kmeans(
        self,
        df: pd.DataFrame,
        params: MvClusterParams
    ) -> MvClusterResult:
        """
        Effectue un clustering K-Means.
        
        Args:
            df: DataFrame avec variables numeriques
            params: Parametres du clustering
            
        Returns:
            MvClusterResult avec labels et centroides
        """
        # Selectionner uniquement les colonnes numeriques et retirer les NA
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        # Standardiser si demande
        if params.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(numeric_df)
        else:
            X = numeric_df.values
        
        # Appliquer K-Means
        kmeans = KMeans(
            n_clusters=params.k,
            max_iter=params.max_iter,
            n_init=params.n_init,
            random_state=42
        )
        labels = kmeans.fit_predict(X)
        
        # Calculer le score de silhouette
        try:
            silhouette = float(silhouette_score(X, labels))
        except:
            silhouette = None
        
        # Informations par cluster
        clusters_info = []
        for cluster_id in range(params.k):
            cluster_mask = labels == cluster_id
            size = int(cluster_mask.sum())
            percentage = float(size / len(labels) * 100)
            
            # Centroide
            centroid = kmeans.cluster_centers_[cluster_id]
            centroid_dict = {
                col: float(centroid[i])
                for i, col in enumerate(numeric_df.columns)
            }
            
            clusters_info.append(
                ClusterInfo(
                    cluster_id=cluster_id,
                    size=size,
                    percentage=percentage,
                    centroid=centroid_dict
                )
            )
        
        return MvClusterResult(
            k=params.k,
            n_points=len(labels),
            inertia=float(kmeans.inertia_),
            silhouette_score=silhouette,
            clusters_info=clusters_info,
            labels=labels.tolist()
        )
    
    def generate_report(
        self,
        df: pd.DataFrame,
        pca_result: MvPcaResult = None,
        cluster_result: MvClusterResult = None
    ) -> MvReportResult:
        """
        Genere un rapport interpretatif.
        
        Args:
            df: DataFrame original
            pca_result: Resultat PCA optionnel
            cluster_result: Resultat clustering optionnel
            
        Returns:
            MvReportResult avec resume
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Informations PCA
        pca_summary = None
        top_pc1 = None
        top_pc2 = None
        
        if pca_result:
            pca_summary = f"PCA avec {pca_result.n_components} composantes expliquant {pca_result.total_variance_explained:.1f}% de la variance"
            
            if len(pca_result.components_info) > 0:
                top_pc1 = [
                    loading.variable
                    for loading in pca_result.components_info[0].top_loadings[:3]
                ]
            
            if len(pca_result.components_info) > 1:
                top_pc2 = [
                    loading.variable
                    for loading in pca_result.components_info[1].top_loadings[:3]
                ]
        
        # Informations clustering
        cluster_summary = None
        cluster_sizes = None
        
        if cluster_result:
            cluster_summary = f"K-Means avec {cluster_result.k} clusters (silhouette={cluster_result.silhouette_score:.3f if cluster_result.silhouette_score else 'N/A'})"
            cluster_sizes = {
                info.cluster_id: info.size
                for info in cluster_result.clusters_info
            }
        
        return MvReportResult(
            dataset_id="",
            n_rows=len(df),
            n_features=len(numeric_df.columns),
            pca_summary=pca_summary,
            top_variables_pc1=top_pc1,
            top_variables_pc2=top_pc2,
            cluster_summary=cluster_summary,
            cluster_sizes=cluster_sizes
        )


# Instance globale du service
mv_service = MvService()
