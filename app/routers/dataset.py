"""
Router pour la generation de datasets.
Endpoint commun a toutes les phases pour generer des donnees synthetiques.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from app.schemas.common import (
    DatasetGenerateRequest,
    DatasetInfo,
    BaseResponse,
    ResponseMeta,
    ResponseReport
)
from app.services.dataset_generator import dataset_generator


router = APIRouter()

def safe_dataframe(df, n=20):
    """
    Convertit un dataframe en JSON-safe dict (max n lignes)
    - remplace NaN par "NA"
    - remplace inf/-inf par 0
    - convertit tout en str
    """
    sample = df.head(n).copy()
    sample = sample.fillna("NA")
    sample = sample.replace([float("inf"), float("-inf")], 0)
    return sample.astype(str).to_dict(orient="records")


@router.post("/generate", response_model=BaseResponse)
async def generate_dataset(request: DatasetGenerateRequest):
    """
    Genere un dataset synthetique pour une phase donnee.
    
    Le dataset est reproductible : meme seed => meme dataset.
    Les donnees generees contiennent des defauts controles selon la phase.
    
    Args:
        request: Requete avec phase, seed et nombre de lignes
        
    Returns:
        Response avec dataset_id et echantillon de donnees
        
    Phases disponibles:
        - clean: Donnees avec missing values, doublons, outliers, types casses
        - eda: Donnees pour analyse exploratoire
        - mv: Donnees pour analyse multivariee (PCA, clustering)
        - ml: Donnees pour apprentissage supervise (classification)
        - ml2: Identique a ml (reutilisation possible)
    """
    try:
        # Generer le dataset
        dataset_id, df, info = dataset_generator.generate(
            phase=request.phase,
            seed=request.seed or 42,
            n=request.n or 1000
        )
        
        # Creer un echantillon (max 20 lignes)
        sample_size = min(20, len(df))
        data_sample = safe_dataframe(df, n=sample_size)
        
        # Creer l'objet DatasetInfo
        dataset_info = DatasetInfo(
            dataset_id=dataset_id,
            phase=info['phase'],
            n_rows=info['n_rows'],
            n_cols=info['n_cols'],
            columns=info['columns'],
            data_sample=data_sample,
            generated_at=info['generated_at']
        )
        
        # Construire la reponse
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=dataset_info.model_dump(),
            report=ResponseReport(
                message=f"Dataset genere avec succes pour la phase '{request.phase}'",
                metrics={
                    "n_rows": info['n_rows'],
                    "n_cols": info['n_cols'],
                    "seed": request.seed or 42
                }
            )
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la generation: {str(e)}")


@router.get("/info/{dataset_id}", response_model=BaseResponse)
async def get_dataset_info(dataset_id: str):
    """
    Recupere les informations sur un dataset genere.
    
    Args:
        dataset_id: Identifiant du dataset
        
    Returns:
        Informations sur le dataset
    """
    try:
        info = dataset_generator.get_dataset_info(dataset_id)
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=info,
            report=ResponseReport(
                message="Informations recuperees avec succes"
            )
        )
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} non trouve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview/{dataset_id}", response_model=BaseResponse)
async def preview_dataset(dataset_id: str, n_rows: int = 20):
    """
    Affiche un apercu d'un dataset.
    
    Args:
        dataset_id: Identifiant du dataset
        n_rows: Nombre de lignes a afficher (max 100)
        
    Returns:
        Echantillon de donnees
    """
    try:
        df = dataset_generator.get_dataset(dataset_id)
        
        # Limiter le nombre de lignes
        n_rows = min(n_rows, 100)
        preview = df.head(n_rows).to_dict('records')
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result={
                "n_rows_total": len(df),
                "n_rows_displayed": len(preview),
                "columns": df.columns.tolist(),
                "data": preview
            },
            report=ResponseReport(
                message=f"Apercu de {len(preview)} lignes sur {len(df)} total"
            )
        )
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} non trouve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
