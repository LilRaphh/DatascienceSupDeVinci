"""
Router pour le TP4 - ML Baseline (phase ml).
Expose les endpoints pour entrainement et prediction.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from app.schemas.common import BaseResponse, ResponseMeta, ResponseReport
from app.schemas.ml import MlTrainParams, MlPredictParams
from app.services.dataset_generator import dataset_generator
from app.services.ml_service import ml_service


router = APIRouter()


@router.post("/train", response_model=BaseResponse)
async def ml_train(
    dataset_id: str,
    params: MlTrainParams
):
    try:
        df = dataset_generator.get_dataset(dataset_id)

        model_id, result = ml_service.train(dataset_id, df, params)

        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"Modele {params.model_type.value} entraine (ID: {model_id})",
                metrics={
                    "accuracy_test": result.metrics_test.accuracy,
                    "f1_test": result.metrics_test.f1_score,
                    "training_time": result.training_time
                }
            )
        )

    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e:
        # pour laisser passer les 400 propres du service (target absente, etc.)
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur entrainement: {str(e)}")


@router.get("/metrics/{model_id}", response_model=BaseResponse)
async def ml_metrics(model_id: str):
    """
    Recupere les metriques d'un modele.
    
    Args:
        model_id: Identifiant du modele
        
    Returns:
        Metriques de performance (accuracy, precision, recall, f1, AUC)
    """
    try:
        metadata = ml_service.get_model_info(model_id)
        
        return BaseResponse(
            meta=ResponseMeta(
                status="success"
            ),
            result=metadata.performance_metrics.model_dump()
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modele {model_id} non trouve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=BaseResponse)
async def ml_predict(
    params: MlPredictParams,
    data: List[Dict[str, Any]]
):
    """
    Fait des predictions avec un modele entraine.
    
    Args:
        params: Parametres avec model_id
        data: Liste de dictionnaires avec les features
        
    Returns:
        Predictions + probabilites pour chaque instance
        
    Exemple de data:
        [
            {"x1": 10.5, "x2": 20.3, "segment": "A"},
            {"x1": 15.2, "x2": 18.7, "segment": "B"}
        ]
    """
    try:
        result = ml_service.predict(params.model_id, data)
        
        return BaseResponse(
            meta=ResponseMeta(
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"{result.n_predictions} prediction(s) effectuee(s)"
            )
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modele {params.model_id} non trouve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prediction: {str(e)}")


@router.get("/model-info/{model_id}", response_model=BaseResponse)
async def ml_model_info(model_id: str):
    """
    Recupere les informations detaillees d'un modele.
    
    Args:
        model_id: Identifiant du modele
        
    Returns:
        Informations completes (hyperparametres, preprocessing, metriques, etc.)
    """
    try:
        info = ml_service.get_model_info(model_id)
        
        return BaseResponse(
            meta=ResponseMeta(
                status="success"
            ),
            result=info.model_dump()
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modele {model_id} non trouve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
