"""
Schemas Pydantic pour le TP5 - ML Avance (phase ml2).
Definit les structures pour le tuning et l'explicabilite.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# ==================== ENUMERATIONS ====================

class SearchStrategy(str, Enum):
    """Strategies de recherche d'hyperparametres."""
    GRID = "grid"
    RANDOM = "random"


# ==================== PARAMETRES ML2 ====================

class Ml2TuneParams(BaseModel):
    """
    Parametres pour l'optimisation d'hyperparametres.
    """
    model_type: str = Field(
        ...,
        description="Type de modele a optimiser (logreg/rf)"
    )
    search: SearchStrategy = Field(
        default=SearchStrategy.GRID,
        description="Strategie de recherche (grid/random)"
    )
    cv: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Nombre de folds pour la validation croisee"
    )
    n_iter: Optional[int] = Field(
        default=10,
        description="Nombre d'iterations pour random search"
    )
    scoring: str = Field(
        default="f1",
        description="Metrique d'optimisation (f1, accuracy, roc_auc, etc.)"
    )


class Ml2PermutationParams(BaseModel):
    """
    Parametres pour l'importance par permutation.
    """
    model_id: str = Field(..., description="Identifiant du modele")
    n_repeats: int = Field(
        default=10,
        ge=5,
        le=30,
        description="Nombre de repetitions pour chaque permutation"
    )
    random_state: int = Field(
        default=42,
        description="Seed pour la reproductibilite"
    )


class Ml2ExplainParams(BaseModel):
    """
    Parametres pour l'explication d'une instance.
    """
    model_id: str = Field(..., description="Identifiant du modele")
    instance: Dict[str, Any] = Field(
        ...,
        description="Instance a expliquer (dictionnaire feature: valeur)"
    )


# ==================== RESULTATS ML2 ====================

class CvResult(BaseModel):
    """
    Resultat d'une configuration d'hyperparametres en CV.
    """
    rank: int = Field(..., description="Rang de cette configuration")
    params: Dict[str, Any] = Field(..., description="Hyperparametres testes")
    mean_score: float = Field(..., description="Score moyen sur les folds")
    std_score: float = Field(..., description="Ecart-type du score")
    scores: List[float] = Field(..., description="Scores sur chaque fold")


class Ml2TuneResult(BaseModel):
    """
    Resultat de l'optimisation d'hyperparametres.
    """
    best_model_id: str = Field(
        ...,
        description="Identifiant du meilleur modele entraine"
    )
    best_params: Dict[str, Any] = Field(
        ...,
        description="Meilleurs hyperparametres trouves"
    )
    best_score: float = Field(
        ...,
        description="Meilleur score obtenu en validation croisee"
    )
    cv_results_summary: List[CvResult] = Field(
        ...,
        description="Resume des 5 meilleures configurations"
    )
    total_fits: int = Field(
        ...,
        description="Nombre total de modeles entraines"
    )
    tuning_time: float = Field(
        ...,
        description="Temps total d'optimisation (secondes)"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Date de creation"
    )


class FeatureImportance(BaseModel):
    """
    Importance d'une feature.
    """
    feature: str = Field(..., description="Nom de la feature")
    importance: float = Field(..., description="Score d'importance")
    rank: int = Field(..., description="Rang d'importance")


class Ml2FeatureImportanceResult(BaseModel):
    """
    Resultat de l'analyse d'importance des features.
    """
    model_id: str = Field(..., description="Identifiant du modele")
    model_type: str = Field(..., description="Type de modele")
    method: str = Field(
        ...,
        description="Methode utilisee (native/permutation)"
    )
    importances: List[FeatureImportance] = Field(
        ...,
        description="Importances triees par ordre decroissant"
    )
    top_features: List[str] = Field(
        ...,
        description="Top 5 features les plus importantes"
    )


class Ml2PermutationResult(BaseModel):
    """
    Resultat de l'importance par permutation.
    """
    model_id: str = Field(..., description="Identifiant du modele")
    n_repeats: int = Field(..., description="Nombre de repetitions utilisees")
    importances: List[FeatureImportance] = Field(
        ...,
        description="Importances calculees par permutation"
    )
    top_features: List[str] = Field(
        ...,
        description="Top 5 features les plus importantes"
    )


class FeatureContribution(BaseModel):
    """
    Contribution d'une feature a une prediction.
    """
    feature: str = Field(..., description="Nom de la feature")
    value: Any = Field(..., description="Valeur de la feature pour cette instance")
    contribution: float = Field(
        ...,
        description="Contribution a la prediction (positif = pousse vers 1)"
    )


class Ml2ExplainResult(BaseModel):
    """
    Explication locale d'une prediction.
    """
    model_id: str = Field(..., description="Identifiant du modele")
    prediction: int = Field(..., description="Classe predite")
    probability: Optional[float] = Field(
        None,
        description="Probabilite de la classe positive"
    )
    contributions: List[FeatureContribution] = Field(
        ...,
        description="Contributions de chaque feature a la prediction"
    )
    top_positive: List[str] = Field(
        ...,
        description="Top 5 features poussant vers la classe 1"
    )
    top_negative: List[str] = Field(
        ...,
        description="Top 5 features poussant vers la classe 0"
    )
    explanation_summary: str = Field(
        ...,
        description="Resume textuel de l'explication"
    )
