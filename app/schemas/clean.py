"""
Schemas Pydantic pour le TP1 - Nettoyage et Preparation (phase clean).
Definit les structures de donnees specifiques au nettoyage.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


# ==================== ENUMERATIONS ====================

class ImputeStrategy(str, Enum):
    """Strategies d'imputation pour les valeurs manquantes."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"


class OutlierStrategy(str, Enum):
    """Strategies de traitement des valeurs aberrantes."""
    CLIP = "clip"
    REMOVE = "remove"
    KEEP = "keep"


class CategoricalStrategy(str, Enum):
    """Strategies d'encodage pour les variables categorielles."""
    ONE_HOT = "one_hot"
    ORDINAL = "ordinal"
    LABEL = "label"


# ==================== PARAMETRES DE NETTOYAGE ====================

class CleanFitParams(BaseModel):
    """
    Parametres pour l'apprentissage du pipeline de nettoyage.
    Definit comment traiter chaque type de probleme.
    """
    impute_strategy: ImputeStrategy = Field(
        default=ImputeStrategy.MEAN,
        description="Strategie d'imputation des valeurs manquantes"
    )
    outlier_strategy: OutlierStrategy = Field(
        default=OutlierStrategy.CLIP,
        description="Strategie de traitement des outliers"
    )
    categorical_strategy: CategoricalStrategy = Field(
        default=CategoricalStrategy.ONE_HOT,
        description="Strategie d'encodage des variables categorielles"
    )
    outlier_threshold: float = Field(
        default=3.0,
        description="Seuil pour la detection d'outliers (en nombre d'ecarts-types)"
    )
    remove_duplicates: bool = Field(
        default=True,
        description="Supprimer les doublons"
    )


class CleanTransformParams(BaseModel):
    """
    Parametres pour appliquer le pipeline de nettoyage.
    Necessite un cleaner_id prealablement appris.
    """
    cleaner_id: str = Field(..., description="Identifiant du pipeline de nettoyage a appliquer")


# ==================== RESULTATS DE NETTOYAGE ====================

class CleaningRules(BaseModel):
    """
    Regles apprises par le pipeline de nettoyage.
    Contient les valeurs utilisees pour l'imputation, le clipping, etc.
    """
    numeric_impute_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Valeurs d'imputation pour chaque colonne numerique"
    )
    categorical_impute_values: Dict[str, str] = Field(
        default_factory=dict,
        description="Valeurs d'imputation pour chaque colonne categorielle"
    )
    outlier_bounds: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Bornes min/max pour le clipping des outliers"
    )
    categorical_mappings: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Mappings pour l'encodage des variables categorielles"
    )


class QualityReport(BaseModel):
    """
    Rapport de qualite des donnees avant nettoyage.
    Identifie tous les problemes detectes.
    """
    n_rows: int = Field(..., description="Nombre total de lignes")
    n_duplicates: int = Field(..., description="Nombre de doublons")
    missing_values: Dict[str, int] = Field(
        default_factory=dict,
        description="Nombre de valeurs manquantes par colonne"
    )
    missing_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Taux de valeurs manquantes par colonne (en %)"
    )
    outliers_count: Dict[str, int] = Field(
        default_factory=dict,
        description="Nombre d'outliers detectes par colonne"
    )
    type_issues: Dict[str, int] = Field(
        default_factory=dict,
        description="Nombre de problemes de type par colonne"
    )
    data_types: Dict[str, str] = Field(
        default_factory=dict,
        description="Types de donnees detectes par colonne"
    )


class CleanFitResult(BaseModel):
    """
    Resultat de l'apprentissage du pipeline de nettoyage.
    Contient l'identifiant du pipeline et les regles apprises.
    """
    cleaner_id: str = Field(..., description="Identifiant unique du pipeline cree")
    dataset_id: str = Field(..., description="Identifiant du dataset d'origine")
    rules: CleaningRules = Field(..., description="Regles de nettoyage apprises")
    quality_before: QualityReport = Field(..., description="Rapport de qualite avant nettoyage")


class CleanTransformResult(BaseModel):
    """
    Resultat de l'application du pipeline de nettoyage.
    Contient les donnees nettoyees et les compteurs de transformations.
    """
    processed_dataset_id: str = Field(..., description="Identifiant du dataset nettoye")
    n_rows_before: int = Field(..., description="Nombre de lignes avant nettoyage")
    n_rows_after: int = Field(..., description="Nombre de lignes apres nettoyage")
    counters: Dict[str, int] = Field(
        default_factory=dict,
        description="Compteurs de transformations (imputations, doublons supprimes, etc.)"
    )
    data_sample: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Echantillon des donnees nettoyees"
    )
