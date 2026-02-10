"""
Service de nettoyage et preparation des donnees (TP1 - phase clean).
Gere l'apprentissage et l'application de pipelines de nettoyage.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import uuid
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from app.schemas.clean import (
    CleanFitParams,
    CleaningRules,
    QualityReport,
    ImputeStrategy,
    OutlierStrategy,
    CategoricalStrategy
)
from app.utils.storage import storage


class CleanService:
    """
    Service de nettoyage de donnees.
    Permet d'apprendre et d'appliquer des transformations de nettoyage.
    """
    
    def __init__(self):
        """Initialisation du service."""
        self.cleaners_cache: Dict[str, Dict[str, Any]] = {}
    
    def generate_quality_report(self, df: pd.DataFrame) -> QualityReport:
        """
        Genere un rapport de qualite des donnees.
        
        Args:
            df: DataFrame a analyser
            
        Returns:
            QualityReport avec toutes les metriques
        """
        # Nombre de doublons
        n_duplicates = df.duplicated().sum()
        
        # Valeurs manquantes
        missing_values = df.isnull().sum().to_dict()
        missing_rates = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Detection des outliers (pour colonnes numeriques)
        outliers_count = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    outliers = ((df[col] - mean).abs() > 3 * std).sum()
                    outliers_count[col] = int(outliers)
                else:
                    outliers_count[col] = 0
        
        # Problemes de types
        type_issues = {}
        for col in df.columns:
            if col in numeric_cols:
                # Verifier si des valeurs non numeriques
                try:
                    pd.to_numeric(df[col], errors='raise')
                    type_issues[col] = 0
                except:
                    non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum() - df[col].isna().sum()
                    type_issues[col] = int(non_numeric)
        
        # Types de donnees
        data_types = df.dtypes.astype(str).to_dict()
        
        return QualityReport(
            n_rows=len(df),
            n_duplicates=int(n_duplicates),
            missing_values={k: int(v) for k, v in missing_values.items()},
            missing_rates={k: float(v) for k, v in missing_rates.items()},
            outliers_count=outliers_count,
            type_issues=type_issues,
            data_types=data_types
        )
    
    def fit(
        self,
        df: pd.DataFrame,
        params: CleanFitParams
    ) -> Tuple[str, CleaningRules, QualityReport]:
        """
        Apprend un pipeline de nettoyage sur les donnees.
        
        Args:
            df: DataFrame a analyser
            params: Parametres de nettoyage
            
        Returns:
            Tuple (cleaner_id, regles_apprises, rapport_qualite)
        """
        # Generer rapport de qualite avant
        quality_before = self.generate_quality_report(df)
        
        # Copier le dataframe pour ne pas modifier l'original
        df_work = df.copy()
        
        # Separer colonnes numeriques et categorielles
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_work.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Nettoyer les types casses dans les colonnes numeriques
        for col in numeric_cols:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
        
        # Calculer les valeurs d'imputation
        numeric_impute_values = {}
        categorical_impute_values = {}
        
        for col in numeric_cols:
            if params.impute_strategy == ImputeStrategy.MEAN:
                numeric_impute_values[col] = float(df_work[col].mean())
            elif params.impute_strategy == ImputeStrategy.MEDIAN:
                numeric_impute_values[col] = float(df_work[col].median())
            elif params.impute_strategy == ImputeStrategy.MODE:
                numeric_impute_values[col] = float(df_work[col].mode()[0]) if len(df_work[col].mode()) > 0 else 0.0
        
        for col in categorical_cols:
            if len(df_work[col].mode()) > 0:
                categorical_impute_values[col] = str(df_work[col].mode()[0])
            else:
                categorical_impute_values[col] = "Unknown"
        
        # Calculer les bornes pour les outliers
        outlier_bounds = {}
        if params.outlier_strategy != OutlierStrategy.KEEP:
            for col in numeric_cols:
                if df_work[col].notna().sum() > 0:
                    mean = df_work[col].mean()
                    std = df_work[col].std()
                    if std > 0:
                        outlier_bounds[col] = {
                            'lower': float(mean - params.outlier_threshold * std),
                            'upper': float(mean + params.outlier_threshold * std)
                        }
        
        # Calculer les mappings pour les categories
        categorical_mappings = {}
        if params.categorical_strategy == CategoricalStrategy.ORDINAL:
            for col in categorical_cols:
                unique_values = df_work[col].dropna().unique()
                categorical_mappings[col] = {
                    val: idx for idx, val in enumerate(sorted(unique_values))
                }
        elif params.categorical_strategy == CategoricalStrategy.LABEL:
            for col in categorical_cols:
                unique_values = df_work[col].dropna().unique()
                categorical_mappings[col] = {
                    val: idx for idx, val in enumerate(unique_values)
                }
        
        # Creer les regles
        rules = CleaningRules(
            numeric_impute_values=numeric_impute_values,
            categorical_impute_values=categorical_impute_values,
            outlier_bounds=outlier_bounds,
            categorical_mappings=categorical_mappings
        )
        
        # Creer un ID unique pour ce cleaner
        cleaner_id = f"cleaner_{uuid.uuid4().hex[:12]}"
        
        # Stocker le cleaner
        cleaner_data = {
            'params': params.model_dump(),
            'rules': rules.model_dump(),
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols
        }
        
        storage.save_model(
            model_id=cleaner_id,
            model=cleaner_data,
            metadata={
                'type': 'cleaner',
                'cleaner_id': cleaner_id,
                'params': params.model_dump(),
                'quality_before': quality_before.model_dump()
            }
        )
        
        self.cleaners_cache[cleaner_id] = cleaner_data
        
        return cleaner_id, rules, quality_before
    
    def transform(
        self,
        df: pd.DataFrame,
        cleaner_id: str
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Applique un pipeline de nettoyage appris.
        
        Args:
            df: DataFrame a nettoyer
            cleaner_id: ID du cleaner a utiliser
            
        Returns:
            Tuple (dataframe_nettoye, compteurs_transformations)
        """
        # Charger le cleaner
        if cleaner_id not in self.cleaners_cache:
            cleaner_data = storage.load_model(cleaner_id)
            self.cleaners_cache[cleaner_id] = cleaner_data
        else:
            cleaner_data = self.cleaners_cache[cleaner_id]
        
        params = CleanFitParams(**cleaner_data['params'])
        rules = CleaningRules(**cleaner_data['rules'])
        numeric_cols = cleaner_data['numeric_cols']
        categorical_cols = cleaner_data['categorical_cols']
        
        # Copier le dataframe
        df_clean = df.copy()
        
        # Compteurs
        counters = {
            'duplicates_removed': 0,
            'missing_imputed_numeric': 0,
            'missing_imputed_categorical': 0,
            'outliers_clipped': 0,
            'outliers_removed': 0,
            'type_errors_fixed': 0
        }
        
        # 1. Supprimer les doublons
        if params.remove_duplicates:
            n_before = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            counters['duplicates_removed'] = n_before - len(df_clean)
            df_clean = df_clean.reset_index(drop=True)
        
        # 2. Fixer les problemes de types dans les colonnes numeriques
        for col in numeric_cols:
            if col in df_clean.columns:
                n_errors = pd.to_numeric(df_clean[col], errors='coerce').isna().sum() - df_clean[col].isna().sum()
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                counters['type_errors_fixed'] += int(n_errors)
        
        # 3. Imputer les valeurs manquantes
        for col in numeric_cols:
            if col in df_clean.columns and col in rules.numeric_impute_values:
                n_missing = df_clean[col].isna().sum()
                if n_missing > 0:
                    df_clean[col].fillna(rules.numeric_impute_values[col], inplace=True)
                    counters['missing_imputed_numeric'] += int(n_missing)
        
        for col in categorical_cols:
            if col in df_clean.columns and col in rules.categorical_impute_values:
                n_missing = df_clean[col].isna().sum()
                if n_missing > 0:
                    df_clean[col].fillna(rules.categorical_impute_values[col], inplace=True)
                    counters['missing_imputed_categorical'] += int(n_missing)
        
        # 4. Traiter les outliers
        if params.outlier_strategy == OutlierStrategy.CLIP:
            for col, bounds in rules.outlier_bounds.items():
                if col in df_clean.columns:
                    n_outliers = ((df_clean[col] < bounds['lower']) | (df_clean[col] > bounds['upper'])).sum()
                    df_clean[col] = df_clean[col].clip(lower=bounds['lower'], upper=bounds['upper'])
                    counters['outliers_clipped'] += int(n_outliers)
        
        elif params.outlier_strategy == OutlierStrategy.REMOVE:
            n_before = len(df_clean)
            for col, bounds in rules.outlier_bounds.items():
                if col in df_clean.columns:
                    df_clean = df_clean[
                        (df_clean[col] >= bounds['lower']) &
                        (df_clean[col] <= bounds['upper'])
                    ]
            counters['outliers_removed'] = n_before - len(df_clean)
            df_clean = df_clean.reset_index(drop=True)
        
        # 5. Encoder les variables categorielles
        if params.categorical_strategy == CategoricalStrategy.ONE_HOT:
            # One-hot encoding
            for col in categorical_cols:
                if col in df_clean.columns:
                    dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
                    df_clean = pd.concat([df_clean, dummies], axis=1)
                    df_clean = df_clean.drop(columns=[col])
        
        elif params.categorical_strategy in [CategoricalStrategy.ORDINAL, CategoricalStrategy.LABEL]:
            # Ordinal/Label encoding
            for col in categorical_cols:
                if col in df_clean.columns and col in rules.categorical_mappings:
                    df_clean[col] = df_clean[col].map(rules.categorical_mappings[col])
        
        return df_clean, counters


# Instance globale du service
clean_service = CleanService()
