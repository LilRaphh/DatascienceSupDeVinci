"""
Schemas Pydantic pour le TP3 - Analyse Multivariee (phase mv).
Definit les structures pour PCA et clustering.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


# ==================== PARAMETRES ANALYSE MULTIVARIEE ====================

class MvPcaParams(BaseModel):
    """
    Parametres pour l'Analyse en Composantes Principales (PCA).
    """
    n_components: int = Field(
        default=2,
        ge=2,
        le=5,
        description="Nombre de composantes principales a calculer (2 a 5)"
    )
    scale: bool = Field(
        default=True,
        description="Standardiser les variables avant PCA"
    )


class MvClusterParams(BaseModel):
    """
    Parametres pour le clustering K-Means.
    """
    k: int = Field(
        default=3,
        ge=2,
        le=6,
        description="Nombre de clusters (2 a 6)"
    )
    scale: bool = Field(
        default=True,
        description="Standardiser les variables avant clustering"
    )
    max_iter: int = Field(
        default=300,
        description="Nombre maximal d'iterations pour K-Means"
    )
    n_init: int = Field(
        default=10,
        description="Nombre d'initialisations differentes"
    )


# ==================== RESULTATS PCA ====================

class PcaLoading(BaseModel):
    """
    Contribution d'une variable a une composante principale.
    """
    variable: str = Field(..., description="Nom de la variable")
    loading: float = Field(..., description="Valeur du loading (contribution)")


class PcaComponent(BaseModel):
    """
    Informations sur une composante principale.
    """
    component: int = Field(..., description="Numero de la composante (1, 2, ...)")
    explained_variance: float = Field(..., description="Variance expliquee par cette composante")
    explained_variance_ratio: float = Field(..., description="% de variance expliquee")
    top_loadings: List[PcaLoading] = Field(
        ...,
        description="Top 5 variables contribuant a cette composante"
    )


class MvPcaResult(BaseModel):
    """
    Resultat de la PCA - projection et informations.
    """
    n_components: int = Field(..., description="Nombre de composantes calculees")
    total_variance_explained: float = Field(
        ...,
        description="Variance totale expliquee par toutes les composantes (%)"
    )
    components_info: List[PcaComponent] = Field(
        ...,
        description="Informations detaillees par composante"
    )
    projection: List[Dict[str, Any]] = Field(
        ...,
        description="Donnees projetees sur les composantes principales (records)"
    )


# ==================== RESULTATS CLUSTERING ====================

class ClusterInfo(BaseModel):
    """
    Informations sur un cluster.
    """
    cluster_id: int = Field(..., description="Identifiant du cluster")
    size: int = Field(..., description="Nombre de points dans ce cluster")
    percentage: float = Field(..., description="Pourcentage du total (%)")
    centroid: Dict[str, float] = Field(
        ...,
        description="Coordonnees du centroide"
    )


class MvClusterResult(BaseModel):
    """
    Resultat du clustering K-Means.
    """
    k: int = Field(..., description="Nombre de clusters")
    n_points: int = Field(..., description="Nombre total de points")
    inertia: float = Field(..., description="Inertie du modele (somme des distances au carre)")
    silhouette_score: Optional[float] = Field(
        None,
        description="Score de silhouette (qualite du clustering)"
    )
    clusters_info: List[ClusterInfo] = Field(
        ...,
        description="Informations sur chaque cluster"
    )
    labels: List[int] = Field(
        ...,
        description="Labels de cluster pour chaque point"
    )


# ==================== RAPPORT MULTIVARIÉ ====================

class MvReportResult(BaseModel):
    """
    Mini-rapport interpretatif sur l'analyse multivariee.
    """
    dataset_id: str = Field(..., description="Identifiant du dataset analyse")
    n_rows: int = Field(..., description="Nombre de lignes")
    n_features: int = Field(..., description="Nombre de variables numeriques")
    
    # Informations PCA si disponibles
    pca_summary: Optional[str] = Field(
        None,
        description="Resume de l'analyse PCA"
    )
    top_variables_pc1: Optional[List[str]] = Field(
        None,
        description="Top 3 variables sur PC1"
    )
    top_variables_pc2: Optional[List[str]] = Field(
        None,
        description="Top 3 variables sur PC2"
    )
    
    # Informations clustering si disponibles
    cluster_summary: Optional[str] = Field(
        None,
        description="Resume du clustering"
    )
    cluster_sizes: Optional[Dict[int, int]] = Field(
        None,
        description="Tailles des clusters"
    )
