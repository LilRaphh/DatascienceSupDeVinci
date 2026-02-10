"""
Router pour le TP2 - EDA (Exploratory Data Analysis).
Expose les endpoints pour statistiques descriptives et graphiques.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.common import BaseResponse, ResponseMeta, ResponseReport
from app.schemas.eda import (
    EdaSummaryParams,
    EdaGroupByParams,
    EdaCorrelationParams,
    EdaPlotsParams
)
from app.services.dataset_generator import dataset_generator
from app.services.eda_service import eda_service


router = APIRouter()


@router.post("/summary", response_model=BaseResponse)
async def eda_summary(
    dataset_id: str,
    params: EdaSummaryParams = EdaSummaryParams()
):
    """
    Calcule les statistiques descriptives pour toutes les variables.
    
    Pour chaque variable numerique:
    - count, mean, std, min, max
    - Quartiles (Q25, Q50/mediane, Q75)
    - Taux de valeurs manquantes
    
    Pour chaque variable categorielle:
    - count, unique, mode, mode_freq
    - Taux de valeurs manquantes
    
    Args:
        dataset_id: Identifiant du dataset
        params: Parametres optionnels (inclure percentiles, etc.)
        
    Returns:
        Statistiques pour toutes les variables
    """
    try:
        # Recuperer le dataset
        df = dataset_generator.get_dataset(dataset_id)
        
        # Calculer les statistiques
        result = eda_service.summary(df)
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"Statistiques calculees pour {result.n_cols} variables sur {result.n_rows} lignes"
            )
        )
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul des statistiques: {str(e)}")


@router.post("/groupby", response_model=BaseResponse)
async def eda_groupby(
    dataset_id: str,
    params: EdaGroupByParams
):
    """
    Effectue un groupby avec agregations sur les variables numeriques.
    
    Permet de calculer des statistiques par groupe:
    - Moyenne (mean)
    - Mediane (median)
    - Somme (sum)
    - Nombre (count)
    - Ecart-type (std)
    - Min/Max
    
    Args:
        dataset_id: Identifiant du dataset
        params: Parametres avec colonne de groupement et metriques
        
    Returns:
        Tableau agrege par groupe
        
    Exemple:
        by="segment", metrics=["mean", "median"]
        => Moyenne et mediane de toutes les variables numeriques par segment
    """
    try:
        # Recuperer le dataset
        df = dataset_generator.get_dataset(dataset_id)
        
        # Effectuer le groupby
        result = eda_service.groupby(df, params)
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"Agregation effectuee par '{params.by}' avec metriques {params.metrics}",
                metrics={
                    "n_groups": len(result.aggregations),
                    "metrics_used": params.metrics
                }
            )
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du groupby: {str(e)}")


@router.post("/correlation", response_model=BaseResponse)
async def eda_correlation(
    dataset_id: str,
    params: EdaCorrelationParams = EdaCorrelationParams()
):
    """
    Calcule la matrice de correlation entre variables numeriques.
    
    Identifie automatiquement les paires de variables les plus correlees
    (en valeur absolue) selon un seuil configurable.
    
    Args:
        dataset_id: Identifiant du dataset
        params: Parametres (methode, seuil)
        
    Returns:
        Matrice de correlation complete + top paires correlees
        
    Methodes disponibles:
        - pearson: Correlation lineaire (par defaut)
        - spearman: Correlation de rang
        - kendall: Tau de Kendall
    """
    try:
        # Recuperer le dataset
        df = dataset_generator.get_dataset(dataset_id)
        
        # Calculer les correlations
        result = eda_service.correlation(df, params)
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"Matrice de correlation calculee avec methode '{params.method}'",
                metrics={
                    "n_variables": len(result.correlation_matrix),
                    "n_strong_correlations": len(result.top_correlations)
                }
            )
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul de correlation: {str(e)}")


@router.post("/plots", response_model=BaseResponse)
async def eda_plots(
    dataset_id: str,
    params: EdaPlotsParams
):
    """
    Genere des graphiques exploratoires.
    
    Types de graphiques disponibles:
    - histogram: Distribution d'une variable numerique
    - boxplot: Boxplot d'une variable numerique par groupe
    - barplot: Distribution d'une variable categorielle
    
    Les graphiques sont retournes au format Plotly JSON dans les artifacts.
    
    Args:
        dataset_id: Identifiant du dataset
        params: Parametres (types de plots, variables)
        
    Returns:
        Graphiques generes dans le champ artifacts
    """
    try:
        # Recuperer le dataset
        df = dataset_generator.get_dataset(dataset_id)
        
        # Generer les graphiques
        result, artifacts = eda_service.plots(df, params)
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            artifacts=artifacts,
            report=ResponseReport(
                message=f"{len(result.plots_generated)} graphique(s) genere(s)",
                metrics={
                    "plots_count": len(result.plots_generated),
                    "plots_types": result.plots_generated
                }
            )
        )
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la generation des graphiques: {str(e)}")
