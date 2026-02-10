"""
Service d'Analyse Exploratoire des Donnees (TP2 - phase eda).
Gere les statistiques descriptives et la generation de graphiques.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import plotly.express as px
import plotly.graph_objects as go

from app.schemas.eda import (
    VariableSummary,
    EdaSummaryResult,
    EdaGroupByResult,
    CorrelationPair,
    EdaCorrelationResult,
    EdaPlotsResult,
    EdaGroupByParams,
    EdaCorrelationParams,
    EdaPlotsParams
)


class EdaService:
    """
    Service d'analyse exploratoire de donnees.
    Fournit des statistiques descriptives et des graphiques.
    """
    
    def summary(self, df: pd.DataFrame) -> EdaSummaryResult:
        """
        Calcule les statistiques descriptives pour toutes les variables.
        
        Args:
            df: DataFrame a analyser
            
        Returns:
            EdaSummaryResult avec statistiques par variable
        """
        summaries = {}
        
        for col in df.columns:
            # Informations de base
            count = int(df[col].notna().sum())
            missing = int(df[col].isna().sum())
            missing_rate = float(missing / len(df) * 100)
            unique = int(df[col].nunique())
            
            if df[col].dtype in ['int64', 'float64']:
                # Variable numerique
                summaries[col] = VariableSummary(
                    count=count,
                    missing=missing,
                    missing_rate=missing_rate,
                    unique=unique,
                    mean=float(df[col].mean()) if count > 0 else None,
                    std=float(df[col].std()) if count > 0 else None,
                    min=float(df[col].min()) if count > 0 else None,
                    max=float(df[col].max()) if count > 0 else None,
                    q25=float(df[col].quantile(0.25)) if count > 0 else None,
                    q50=float(df[col].quantile(0.50)) if count > 0 else None,
                    q75=float(df[col].quantile(0.75)) if count > 0 else None
                )
            else:
                # Variable categorielle
                mode_value = df[col].mode()
                summaries[col] = VariableSummary(
                    count=count,
                    missing=missing,
                    missing_rate=missing_rate,
                    unique=unique,
                    mode=str(mode_value[0]) if len(mode_value) > 0 else None,
                    mode_freq=int(df[col].value_counts().iloc[0]) if count > 0 else None
                )
        
        return EdaSummaryResult(
            n_rows=len(df),
            n_cols=len(df.columns),
            summaries=summaries
        )
    
    def groupby(
        self,
        df: pd.DataFrame,
        params: EdaGroupByParams
    ) -> EdaGroupByResult:
        """
        Effectue un groupby avec agregations.
        
        Args:
            df: DataFrame a analyser
            params: Parametres de groupement
            
        Returns:
            EdaGroupByResult avec agregations
        """
        # Verifier que la colonne de groupement existe
        if params.by not in df.columns:
            raise ValueError(f"Colonne {params.by} non trouvee")
        
        # Selectionner les colonnes numeriques si non specifiees
        if params.columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in params.columns if col in df.columns]
        
        # Effectuer les agregations
        grouped = df.groupby(params.by)
        
        results = []
        for name, group in grouped:
            row = {params.by: name}
            
            for col in numeric_cols:
                if col in group.columns:
                    for metric in params.metrics:
                        if metric == "mean":
                            row[f"{col}_mean"] = float(group[col].mean())
                        elif metric == "median":
                            row[f"{col}_median"] = float(group[col].median())
                        elif metric == "sum":
                            row[f"{col}_sum"] = float(group[col].sum())
                        elif metric == "count":
                            row[f"{col}_count"] = int(group[col].count())
                        elif metric == "std":
                            row[f"{col}_std"] = float(group[col].std())
                        elif metric == "min":
                            row[f"{col}_min"] = float(group[col].min())
                        elif metric == "max":
                            row[f"{col}_max"] = float(group[col].max())
            
            results.append(row)
        
        return EdaGroupByResult(
            grouped_by=params.by,
            metrics_used=params.metrics,
            aggregations=results
        )
    
    def correlation(
        self,
        df: pd.DataFrame,
        params: EdaCorrelationParams
    ) -> EdaCorrelationResult:
        """
        Calcule la matrice de correlation et identifie les top correlations.
        
        Args:
            df: DataFrame a analyser
            params: Parametres de correlation
            
        Returns:
            EdaCorrelationResult avec matrice et top paires
        """
        # Selectionner uniquement les colonnes numeriques
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            raise ValueError("Au moins 2 variables numeriques sont necessaires")
        
        # Calculer la matrice de correlation
        corr_matrix = numeric_df.corr(method=params.method)
        
        # Convertir en dictionnaire
        corr_dict = corr_matrix.to_dict()
        
        # Identifier les top correlations (en valeur absolue)
        top_correlations = []
        cols = corr_matrix.columns.tolist()
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= params.threshold:
                    top_correlations.append(
                        CorrelationPair(
                            var1=cols[i],
                            var2=cols[j],
                            correlation=float(corr_value)
                        )
                    )
        
        # Trier par valeur absolue decroissante
        top_correlations.sort(key=lambda x: abs(x.correlation), reverse=True)
        
        return EdaCorrelationResult(
            method=params.method,
            correlation_matrix=corr_dict,
            top_correlations=top_correlations
        )
    
    def plots(
        self,
        df: pd.DataFrame,
        params: EdaPlotsParams
    ) -> Tuple[EdaPlotsResult, Dict[str, Any]]:
        """
        Genere des graphiques exploratoires.
        
        Args:
            df: DataFrame a analyser
            params: Parametres de visualisation
            
        Returns:
            Tuple (EdaPlotsResult, artifacts_dict) avec les graphiques en JSON
        """
        artifacts = {}
        plots_generated = []
        
        # Histogram pour variable numerique
        if "histogram" in params.plot_types:
            if params.numeric_var and params.numeric_var in df.columns:
                fig = px.histogram(
                    df,
                    x=params.numeric_var,
                    nbins=30,
                    title=f"Distribution de {params.numeric_var}"
                )
                artifacts["histogram"] = fig.to_json()
                plots_generated.append("histogram")
        
        # Boxplot par groupe
        if "boxplot" in params.plot_types:
            if params.numeric_var and params.group_by:
                if params.numeric_var in df.columns and params.group_by in df.columns:
                    fig = px.box(
                        df,
                        x=params.group_by,
                        y=params.numeric_var,
                        title=f"{params.numeric_var} par {params.group_by}"
                    )
                    artifacts["boxplot"] = fig.to_json()
                    plots_generated.append("boxplot")
        
        # Barplot pour variable categorielle
        if "barplot" in params.plot_types:
            if params.categorical_var and params.categorical_var in df.columns:
                value_counts = df[params.categorical_var].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    labels={'x': params.categorical_var, 'y': 'Nombre'},
                    title=f"Distribution de {params.categorical_var}"
                )
                artifacts["barplot"] = fig.to_json()
                plots_generated.append("barplot")
        
        result = EdaPlotsResult(plots_generated=plots_generated)
        
        return result, artifacts


# Instance globale du service
eda_service = EdaService()
