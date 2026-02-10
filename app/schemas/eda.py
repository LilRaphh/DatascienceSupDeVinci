"""
Schemas Pydantic pour le TP2 - EDA (Exploratory Data Analysis).
Definit les structures pour les statistiques descriptives et graphiques.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


# ==================== PARAMETRES EDA ====================

class EdaSummaryParams(BaseModel):
    """
    Parametres pour les statistiques descriptives.
    """
    include_percentiles: bool = Field(
        default=True,
        description="Inclure les percentiles (25%, 50%, 75%)"
    )


class EdaGroupByParams(BaseModel):
    """
    Parametres pour le groupby avec agregation.
    """
    by: str = Field(..., description="Colonne de groupement (ex: 'segment')")
    metrics: List[str] = Field(
        default=["mean", "median"],
        description="Metriques d'agregation a calculer"
    )
    columns: Optional[List[str]] = Field(
        None,
        description="Colonnes a agreger (si None, toutes les numeriques)"
    )


class EdaCorrelationParams(BaseModel):
    """
    Parametres pour la matrice de correlation.
    """
    method: str = Field(
        default="pearson",
        description="Methode de correlation (pearson/spearman/kendall)"
    )
    threshold: float = Field(
        default=0.5,
        description="Seuil pour identifier les top correlations"
    )


class EdaPlotsParams(BaseModel):
    """
    Parametres pour la generation de graphiques.
    """
    plot_types: List[str] = Field(
        default=["histogram", "boxplot", "barplot"],
        description="Types de graphiques a generer"
    )
    numeric_var: Optional[str] = Field(
        None,
        description="Variable numerique pour histogram/boxplot"
    )
    categorical_var: Optional[str] = Field(
        None,
        description="Variable categorielle pour barplot"
    )
    group_by: Optional[str] = Field(
        None,
        description="Variable de groupement pour boxplot"
    )


# ==================== RESULTATS EDA ====================

class VariableSummary(BaseModel):
    """
    Statistiques descriptives pour une variable.
    """
    count: int = Field(..., description="Nombre de valeurs non-nulles")
    missing: int = Field(..., description="Nombre de valeurs manquantes")
    missing_rate: float = Field(..., description="Taux de valeurs manquantes (%)")
    unique: Optional[int] = Field(None, description="Nombre de valeurs uniques")
    
    # Pour les variables numeriques
    mean: Optional[float] = Field(None, description="Moyenne")
    std: Optional[float] = Field(None, description="Ecart-type")
    min: Optional[float] = Field(None, description="Minimum")
    max: Optional[float] = Field(None, description="Maximum")
    q25: Optional[float] = Field(None, description="1er quartile (25%)")
    q50: Optional[float] = Field(None, description="Mediane (50%)")
    q75: Optional[float] = Field(None, description="3eme quartile (75%)")
    
    # Pour les variables categorielles
    mode: Optional[str] = Field(None, description="Valeur la plus frequente")
    mode_freq: Optional[int] = Field(None, description="Frequence du mode")


class EdaSummaryResult(BaseModel):
    """
    Resultat du summary - statistiques pour toutes les variables.
    """
    n_rows: int = Field(..., description="Nombre total de lignes")
    n_cols: int = Field(..., description="Nombre total de colonnes")
    summaries: Dict[str, VariableSummary] = Field(
        ...,
        description="Statistiques par variable"
    )


class EdaGroupByResult(BaseModel):
    """
    Resultat du groupby - agregations par groupe.
    """
    grouped_by: str = Field(..., description="Variable de groupement utilisee")
    metrics_used: List[str] = Field(..., description="Metriques calculees")
    aggregations: List[Dict[str, Any]] = Field(
        ...,
        description="Resultats agreges (format records)"
    )


class CorrelationPair(BaseModel):
    """
    Paire de variables avec leur correlation.
    """
    var1: str = Field(..., description="Premiere variable")
    var2: str = Field(..., description="Deuxieme variable")
    correlation: float = Field(..., description="Coefficient de correlation")


class EdaCorrelationResult(BaseModel):
    """
    Resultat de l'analyse de correlation.
    """
    method: str = Field(..., description="Methode utilisee")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Matrice de correlation complete"
    )
    top_correlations: List[CorrelationPair] = Field(
        ...,
        description="Top paires de variables correlees"
    )


class EdaPlotsResult(BaseModel):
    """
    Resultat de la generation de graphiques.
    Les graphiques sont retournes en format Plotly JSON.
    """
    plots_generated: List[str] = Field(
        ...,
        description="Liste des types de graphiques generes"
    )
