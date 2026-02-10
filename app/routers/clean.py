"""
Router pour le TP1 - Nettoyage et Preparation (phase clean).
Gere l'apprentissage et l'application de pipelines de nettoyage.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.schemas.common import BaseResponse, ResponseMeta, ResponseReport
from app.schemas.clean import (
    CleanFitParams,
    CleanTransformParams,
    CleanFitResult,
    CleanTransformResult
)
from app.services.dataset_generator import dataset_generator
from app.services.clean_service import clean_service


router = APIRouter()


@router.post("/fit", response_model=BaseResponse)
async def clean_fit(
    dataset_id: str,
    params: CleanFitParams
):
    """
    Apprend un pipeline de nettoyage sur les donnees brutes.
    
    Le pipeline apprend les regles de nettoyage a partir des donnees :
    - Valeurs d'imputation pour les NA
    - Bornes pour le clipping des outliers
    - Encodages pour les variables categorielles
    
    Args:
        dataset_id: Identifiant du dataset a nettoyer
        params: Parametres de nettoyage (strategies d'imputation, outliers, etc.)
        
    Returns:
        cleaner_id et regles apprises + rapport qualite avant nettoyage
    """
    try:
        # Recuperer le dataset
        df = dataset_generator.get_dataset(dataset_id)
        
        # Apprendre le pipeline de nettoyage
        cleaner_id, rules, quality_before = clean_service.fit(df, params)
        
        # Construire le resultat
        result = CleanFitResult(
            cleaner_id=cleaner_id,
            dataset_id=dataset_id,
            rules=rules,
            quality_before=quality_before
        )
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message=f"Pipeline de nettoyage cree avec succes (ID: {cleaner_id})",
                metrics={
                    "n_rows": quality_before.n_rows,
                    "n_duplicates": quality_before.n_duplicates,
                    "total_missing": sum(quality_before.missing_values.values()),
                    "total_outliers": sum(quality_before.outliers_count.values())
                }
            )
        )
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'apprentissage: {str(e)}")


@router.post("/transform", response_model=BaseResponse)
async def clean_transform(
    dataset_id: str,
    params: CleanTransformParams
):
    """
    Applique un pipeline de nettoyage appris aux donnees.
    
    Transformations appliquees :
    - Suppression des doublons
    - Imputation des valeurs manquantes
    - Traitement des outliers (clip ou remove)
    - Correction des problemes de types
    - Encodage des variables categorielles
    
    Args:
        dataset_id: Identifiant du dataset a transformer
        params: Parametres avec cleaner_id
        
    Returns:
        Donnees nettoyees + compteurs des transformations effectuees
    """
    try:
        # Recuperer le dataset
        df = dataset_generator.get_dataset(dataset_id)
        
        # Appliquer le nettoyage
        df_clean, counters = clean_service.transform(df, params.cleaner_id)
        
        # Creer un nouvel ID pour le dataset nettoye
        import uuid
        processed_dataset_id = f"clean_{params.cleaner_id}_{uuid.uuid4().hex[:8]}"
        
        # Stocker le dataset nettoye (en option, ici on le retourne juste)
        # On pourrait aussi le stocker dans le cache du generateur
        
        # Creer un echantillon
        sample_size = min(20, len(df_clean))
        data_sample = df_clean.head(sample_size).to_dict('records')
        
        # Construire le resultat
        result = CleanTransformResult(
            processed_dataset_id=processed_dataset_id,
            n_rows_before=len(df),
            n_rows_after=len(df_clean),
            counters=counters,
            data_sample=data_sample
        )
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=result.model_dump(),
            report=ResponseReport(
                message="Nettoyage applique avec succes",
                metrics={
                    "rows_removed": len(df) - len(df_clean),
                    "total_imputations": counters.get('missing_imputed_numeric', 0) + counters.get('missing_imputed_categorical', 0),
                    "duplicates_removed": counters.get('duplicates_removed', 0)
                }
            )
        )
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Cleaner {params.cleaner_id} non trouve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transformation: {str(e)}")


@router.get("/report/{dataset_id}", response_model=BaseResponse)
async def clean_report(dataset_id: str):
    """
    Genere un rapport de qualite des donnees sans transformation.
    
    Identifie tous les problemes dans les donnees :
    - Valeurs manquantes (nombre et taux)
    - Doublons
    - Outliers detectes
    - Problemes de types de donnees
    
    Args:
        dataset_id: Identifiant du dataset a analyser
        
    Returns:
        Rapport detaille de qualite
    """
    try:
        # Recuperer le dataset
        df = dataset_generator.get_dataset(dataset_id)
        
        # Generer le rapport de qualite
        quality_report = clean_service.generate_quality_report(df)
        
        return BaseResponse(
            meta=ResponseMeta(
                dataset_id=dataset_id,
                status="success"
            ),
            result=quality_report.model_dump(),
            report=ResponseReport(
                message="Rapport de qualite genere avec succes",
                warnings=[
                    f"Doublons detectes: {quality_report.n_duplicates}",
                    f"Valeurs manquantes totales: {sum(quality_report.missing_values.values())}",
                    f"Outliers detectes: {sum(quality_report.outliers_count.values())}"
                ] if (quality_report.n_duplicates > 0 or 
                      sum(quality_report.missing_values.values()) > 0 or 
                      sum(quality_report.outliers_count.values()) > 0) else []
            )
        )
    
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la generation du rapport: {str(e)}")
