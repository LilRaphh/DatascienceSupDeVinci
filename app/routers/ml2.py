"""
Router pour le TP5 - ML Avance (phase ml2).
Expose les endpoints pour tuning et explicabilite.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.schemas.common import BaseResponse, ResponseMeta, ResponseReport
from app.schemas.ml2 import *
from app.services.dataset_generator import dataset_generator
from app.services.ml2_service import ml2_service


router = APIRouter()


@router.post("/tune", response_model=BaseResponse)
async def ml2_tune(
    dataset_id: str,
    params: Ml2TuneParams
):
    """
    Optimise les hyperparametres avec validation croisee.
    
    Strategies disponibles:
    - grid: Grid Search (teste toutes les combinaisons)
    - random: Random Search (echantillonne n_iter combinaisons)
    
    Le meilleur modele est automatiquement sauvegarde.
    
    Args:
        dataset_id: Identifiant du dataset
        params: Parametres (model_type, search, cv, scoring)
        
    Returns:
        best_model_id + best_params + resume des 5 meilleures configs
    """
    try:
        df = dataset_generator.get_dataset(dataset_id)
        
        result = ml2_service.tune(df, params)
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"Optimisation terminee. Meilleur modele: {result.best_model_id}",
                metrics={
                    "best_score": result.best_score,
                    "total_fits": result.total_fits,
                    "tuning_time": result.tuning_time
                }
            )
        )
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur tuning: {str(e)}")


@router.get("/feature-importance/{model_id}", response_model=BaseResponse)
async def ml2_feature_importance(model_id: str):
    """
    Calcule l'importance native des features.
    
    - Random Forest: importance native basee sur la reduction d'impurete
    - Logistic Regression: valeurs absolues des coefficients
    
    Args:
        model_id: Identifiant du modele
        
    Returns:
        Importance triee par ordre decroissant + top 5 features
    """
    try:
        result = ml2_service.feature_importance(model_id)
        
        return BaseResponse(
            meta=ResponseMeta(
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"Importance calculee avec methode '{result.method}'",
                metrics={
                    "top_feature": result.top_features[0] if result.top_features else None
                }
            )
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modele {model_id} non trouve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/permutation-importance", response_model=BaseResponse)
async def ml2_permutation_importance(
    dataset_id: str,
    params: Ml2PermutationParams
):
    """
    Calcule l'importance par permutation (methode modele-agnostique).
    
    Principe: permuter chaque feature et mesurer la degradation de performance.
    Plus la degradation est importante, plus la feature est importante.
    
    Args:
        dataset_id: Identifiant du dataset
        params: Parametres (model_id, n_repeats)
        
    Returns:
        Importance par permutation + top 5 features
    """
    try:
        df = dataset_generator.get_dataset(dataset_id)
        
        result = ml2_service.permutation_importance_analysis(
            params.model_id, df, params
        )
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"Importance par permutation calculee ({params.n_repeats} repetitions)",
                metrics={
                    "n_repeats": params.n_repeats
                }
            )
        )
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/explain-instance", response_model=BaseResponse)
async def ml2_explain_instance(
    params: Ml2ExplainParams
):
    """
    Explique la prediction pour une instance specifique.
    
    Fournit une explication locale:
    - Contribution de chaque feature a la prediction
    - Top 5 facteurs poussant vers classe 1
    - Top 5 facteurs poussant vers classe 0
    
    Args:
        params: Parametres (model_id, instance)
        
    Returns:
        Explication locale avec contributions + resume textuel
        
    Exemple d'instance:
        {"x1": 15.2, "x2": 22.5, "x3": 8.1, "segment": "A"}
    """
    try:
        result = ml2_service.explain_instance(
            params.model_id,
            params.instance
        )
        
        return BaseResponse(
            meta=ResponseMeta(
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=result.explanation_summary
            )
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modele {params.model_id} non trouve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
