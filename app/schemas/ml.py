"""
Schemas Pydantic pour le TP4 - ML Baseline (phase ml).
Definit les structures pour l'entrainement et la prediction de modeles.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# ==================== ENUMERATIONS ====================

class ModelType(str, Enum):
    """Types de modeles disponibles."""
    LOGREG = "logreg"
    RF = "rf"


# ==================== PARAMETRES ML ====================

class MlTrainParams(BaseModel):
    """
    Parametres pour l'entrainement d'un modele.
    """
    model_type: ModelType = Field(
        ...,
        description="Type de modele a entrainer (logreg/rf)"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.4,
        description="Proportion du jeu de test (0.1 a 0.4)"
    )
    random_state: int = Field(
        default=42,
        description="Seed pour la reproductibilite du split"
    )
    
    # Hyperparametres optionnels pour Logistic Regression
    logreg_C: float = Field(
        default=1.0,
        description="Parametre de regularisation pour LogReg"
    )
    logreg_max_iter: int = Field(
        default=1000,
        description="Nombre maximal d'iterations pour LogReg"
    )
    
    # Hyperparametres optionnels pour Random Forest
    rf_n_estimators: int = Field(
        default=100,
        description="Nombre d'arbres pour Random Forest"
    )
    rf_max_depth: Optional[int] = Field(
        default=None,
        description="Profondeur maximale des arbres"
    )


class MlPredictParams(BaseModel):
    """
    Parametres pour faire des predictions avec un modele.
    """
    model_id: str = Field(..., description="Identifiant du modele a utiliser")


# ==================== RESULTATS ML ====================

class MlMetrics(BaseModel):
    """
    Metriques d'evaluation d'un modele de classification binaire.
    """
    accuracy: float = Field(..., description="Taux de precision globale")
    precision: float = Field(..., description="Precision (VP / (VP + FP))")
    recall: float = Field(..., description="Rappel (VP / (VP + FN))")
    f1_score: float = Field(..., description="F1-score (moyenne harmonique precision/recall)")
    auc_roc: Optional[float] = Field(None, description="Aire sous la courbe ROC")
    
    # Matrice de confusion
    confusion_matrix: Optional[List[List[int]]] = Field(
        None,
        description="Matrice de confusion [[TN, FP], [FN, TP]]"
    )


class MlTrainResult(BaseModel):
    """
    Resultat de l'entrainement d'un modele.
    """
    model_id: str = Field(..., description="Identifiant unique du modele entraine")
    model_type: str = Field(..., description="Type de modele (logreg/rf)")
    dataset_id: str = Field(..., description="Identifiant du dataset utilise")
    
    # Informations sur les donnees
    n_train: int = Field(..., description="Nombre d'exemples d'entrainement")
    n_test: int = Field(..., description="Nombre d'exemples de test")
    features_used: List[str] = Field(..., description="Liste des features utilisees")
    
    # Metriques
    metrics_train: MlMetrics = Field(..., description="Metriques sur l'ensemble d'entrainement")
    metrics_test: MlMetrics = Field(..., description="Metriques sur l'ensemble de test")
    
    # Informations complementaires
    training_time: float = Field(..., description="Temps d'entrainement en secondes")
    created_at: datetime = Field(default_factory=datetime.now, description="Date de creation")


class MlPrediction(BaseModel):
    """
    Prediction pour une instance.
    """
    prediction: int = Field(..., description="Classe predite (0 ou 1)")
    probability: Optional[List[float]] = Field(
        None,
        description="Probabilites pour chaque classe [P(0), P(1)]"
    )


class MlPredictResult(BaseModel):
    """
    Resultat des predictions.
    """
    model_id: str = Field(..., description="Identifiant du modele utilise")
    n_predictions: int = Field(..., description="Nombre de predictions effectuees")
    predictions: List[MlPrediction] = Field(..., description="Liste des predictions")


class MlModelInfo(BaseModel):
    """
    Informations detaillees sur un modele sauvegarde.
    """
    model_id: str = Field(..., description="Identifiant du modele")
    model_type: str = Field(..., description="Type de modele")
    dataset_id: str = Field(..., description="Dataset d'entrainement")
    
    # Hyperparametres
    hyperparameters: Dict[str, Any] = Field(
        ...,
        description="Hyperparametres utilises"
    )
    
    # Preprocessing applique
    preprocessing_steps: List[str] = Field(
        default_factory=list,
        description="Etapes de preprocessing appliquees"
    )
    features_used: List[str] = Field(..., description="Features utilisees")
    
    # Metriques de performance
    performance_metrics: MlMetrics = Field(
        ...,
        description="Metriques sur l'ensemble de test"
    )
    
    # Metadonnees
    created_at: datetime = Field(..., description="Date de creation")
    training_time: float = Field(..., description="Temps d'entrainement (secondes)")
