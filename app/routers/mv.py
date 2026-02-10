"""
Router pour le TP3 - Analyse Multivariee (phase mv).
Expose les endpoints pour PCA et clustering.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.common import BaseResponse, ResponseMeta, ResponseReport
from app.schemas.mv import MvPcaParams, MvClusterParams
from app.services.dataset_generator import dataset_generator
from app.services.mv_service import mv_service


router = APIRouter()


@router.post("/pca/fit_transform", response_model=BaseResponse)
async def mv_pca_fit_transform(
    dataset_id: str,
    params: MvPcaParams = MvPcaParams()
):
    """
    Effectue une PCA et projette les donnees.
    
    La PCA (Principal Component Analysis) reduit la dimensionnalite
    en trouvant les axes de variance maximale.
    
    Args:
        dataset_id: Identifiant du dataset
        params: Parametres PCA (n_components, scale)
        
    Returns:
        - Projection des donnees sur les composantes principales
        - Variance expliquee par chaque composante
        - Loadings (contributions des variables)
        
    Exemple d'interpretation:
        - PC1 explique 45% de la variance
        - Top variables PC1: x1 (0.8), x5 (0.7), x2 (0.6)
        => PC1 represente principalement x1 et x5
    """
    try:
        df = dataset_generator.get_dataset(dataset_id)
        
        result = mv_service.pca_fit_transform(df, params)
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"PCA effectuee avec {params.n_components} composantes",
                metrics={
                    "variance_expliquee": result.total_variance_explained,
                    "n_points": len(result.projection)
                }
            )
        )
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur PCA: {str(e)}")


@router.post("/cluster/kmeans", response_model=BaseResponse)
async def mv_cluster_kmeans(
    dataset_id: str,
    params: MvClusterParams = MvClusterParams()
):
    """
    Effectue un clustering K-Means.
    
    K-Means regroupe les points en k clusters en minimisant
    la variance intra-cluster.
    
    Args:
        dataset_id: Identifiant du dataset
        params: Parametres clustering (k, scale, max_iter, n_init)
        
    Returns:
        - Labels de cluster pour chaque point
        - Centroides de chaque cluster
        - Score de silhouette (qualite du clustering)
        - Informations par cluster (taille, pourcentage)
        
    Interpretation silhouette:
        - > 0.7: Excellente separation
        - 0.5-0.7: Bonne separation
        - 0.25-0.5: Separation faible
        - < 0.25: Pas de structure claire
    """
    try:
        df = dataset_generator.get_dataset(dataset_id)
        
        result = mv_service.cluster_kmeans(df, params)
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"Clustering K-Means avec {params.k} clusters",
                metrics={
                    "silhouette_score": result.silhouette_score,
                    "inertia": result.inertia,
                    "n_points": result.n_points
                }
            )
        )
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur clustering: {str(e)}")


@router.get("/report/{dataset_id}", response_model=BaseResponse)
async def mv_report(dataset_id: str):
    """
    Genere un mini-rapport interpretatif.
    
    Fournit un resume textuel de l'analyse multivariee:
    - Top variables sur les composantes principales
    - Tailles et repartition des clusters
    
    Args:
        dataset_id: Identifiant du dataset
        
    Returns:
        Rapport interpretatif
        
    Note: Pour obtenir un rapport complet, effectuez d'abord
          une PCA et/ou un clustering puis consultez ce rapport.
    """
    try:
        df = dataset_generator.get_dataset(dataset_id)
        
        # Generer un rapport de base
        report = mv_service.generate_report(df)
        report.dataset_id = dataset_id
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=report.model_dump(),
            report=ResponseReport(
                message="Rapport genere",
                metrics={
                    "n_features": report.n_features
                }
            )
        )
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur rapport: {str(e)}")
