"""
Schemas Pydantic communs utilises par tous les endpoints.
Definit l'enveloppe standard pour les requetes et reponses.
"""

from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
from datetime import datetime


# ==================== SCHEMAS DE REQUETE ====================

class RequestMeta(BaseModel):
    """
    Metadonnees d'une requete.
    Contient les identifiants et versions pour la tracabilite.
    """
    dataset_id: Optional[str] = Field(None, description="Identifiant du dataset a utiliser")
    schema_version: str = Field(default="1.0", description="Version du schema utilise")


class RequestData(BaseModel):
    """
    Donnees brutes d'une requete.
    Peut contenir des records (lignes de donnees).
    """
    records: Optional[List[Dict[str, Any]]] = Field(None, description="Lignes de donnees")


class BaseRequest(BaseModel):
    """
    Structure standard d'une requete.
    Toutes les requetes suivent ce format de base.
    """
    meta: RequestMeta = Field(default_factory=RequestMeta)
    data: Optional[RequestData] = Field(None, description="Donnees optionnelles")
    params: Optional[Dict[str, Any]] = Field(None, description="Parametres specifiques a l'endpoint")


# ==================== SCHEMAS DE REPONSE ====================

class ResponseMeta(BaseModel):
    """
    Metadonnees d'une reponse.
    Contient les informations de tracabilite et de timing.
    """
    dataset_id: Optional[str] = Field(None, description="Identifiant du dataset utilise")
    timestamp: datetime = Field(default_factory=datetime.now, description="Horodatage de la reponse")
    schema_version: str = Field(default="1.0", description="Version du schema")
    status: str = Field(default="success", description="Statut de l'operation (success/error)")


class ResponseReport(BaseModel):
    """
    Rapport d'execution d'une operation.
    Contient les informations sur le traitement effectue.
    """
    message: Optional[str] = Field(None, description="Message descriptif de l'operation")
    warnings: List[str] = Field(default_factory=list, description="Avertissements eventuels")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Metriques de l'operation")


class BaseResponse(BaseModel):
    """
    Structure standard d'une reponse.
    Toutes les reponses suivent ce format de base.
    """
    meta: ResponseMeta
    result: Optional[Any] = Field(None, description="Resultat principal de l'operation")
    report: Optional[ResponseReport] = Field(None, description="Rapport d'execution")
    artifacts: Optional[Dict[str, Any]] = Field(None, description="Artefacts generes (graphiques, etc.)")


# ==================== SCHEMAS POUR LA GENERATION DE DATASETS ====================

class DatasetGenerateRequest(BaseModel):
    """
    Requete pour generer un dataset.
    Permet de specifier la phase, le seed et le nombre de lignes.
    """
    phase: str = Field("clear", description="Phase du projet (clean/eda/mv/ml/ml2)")
    seed: Optional[int] = Field(42, description="Seed pour la reproductibilite")
    n: Optional[int] = Field(1000, description="Nombre de lignes a generer")

    
    class Config:
        schema_extra = {
            "example": {
                "phase": "clean",
                "seed": 123,
                "n": 500
            }
        }

class DatasetInfo(BaseModel):
    """
    Informations sur un dataset genere.
    Contient les metadonnees et un echantillon des donnees.
    """
    dataset_id: str = Field(..., description="Identifiant unique du dataset")
    phase: str = Field(..., description="Phase du projet")
    n_rows: int = Field(..., description="Nombre de lignes")
    n_cols: int = Field(..., description="Nombre de colonnes")
    columns: List[str] = Field(..., description="Noms des colonnes")
    data_sample: List[Dict[str, Any]] = Field(..., description="Echantillon de donnees (max 20 lignes)")
    generated_at: datetime = Field(default_factory=datetime.now, description="Date de generation")
