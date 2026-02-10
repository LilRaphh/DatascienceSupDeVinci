"""
Service de generation de datasets pour les differentes phases.
Genere des donnees synthetiques avec des defauts controles.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import uuid
from datetime import datetime


class DatasetGenerator:
    """
    Generateur de datasets pour chaque phase du projet.
    Permet de generer des donnees reproductibles avec des defauts injectes.
    """
    
    def __init__(self):
        """Initialisation du generateur."""
        self.datasets_cache: Dict[str, pd.DataFrame] = {}
        self.datasets_info: Dict[str, Dict[str, Any]] = {}
    
    def generate(
        self,
        phase: str,
        seed: int = 42,
        n: int = 1000
    ) -> tuple[str, pd.DataFrame, Dict[str, Any]]:
        """
        Genere un dataset pour une phase specifique.
        
        Args:
            phase: Phase du projet (clean/eda/mv/ml/ml2)
            seed: Seed pour la reproductibilite
            n: Nombre de lignes a generer
            
        Returns:
            Tuple (dataset_id, dataframe, info_dict)
        """
        # Definir le seed pour la reproductibilite
        np.random.seed(seed)
        
        # Generer selon la phase
        if phase == "clean":
            df = self._generate_clean_data(n)
        elif phase == "eda":
            df = self._generate_eda_data(n)
        elif phase == "mv":
            df = self._generate_mv_data(n)
        elif phase in ["ml", "ml2"]:
            df = self._generate_ml_data(n)
        else:
            raise ValueError(f"Phase inconnue: {phase}")
        
        # Creer un identifiant unique
        dataset_id = f"{phase}_{seed}_{n}_{uuid.uuid4().hex[:8]}"
        
        # Stocker en cache
        self.datasets_cache[dataset_id] = df
        
        # Creer les infos
        info = {
            "dataset_id": dataset_id,
            "phase": phase,
            "seed": seed,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": df.columns.tolist(),
            "generated_at": datetime.now()
        }
        self.datasets_info[dataset_id] = info
        
        return dataset_id, df, info
    
    def _generate_clean_data(self, n: int) -> pd.DataFrame:
        """
        Genere des donnees pour la phase clean avec defauts.
        
        Defauts injectes:
        - Missing values: 10-20% selon colonnes
        - Doublons: 1-5%
        - Outliers: 1-3% extremes
        - Types casses: quelques valeurs non numeriques dans x2
        """
        # Generer les donnees de base
        df = pd.DataFrame({
            'x1': np.random.normal(50, 15, n),
            'x2': np.random.normal(100, 30, n),
            'x3': np.random.exponential(10, n),
            'segment': np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2]),
            'target': np.random.choice([0, 1], n, p=[0.6, 0.4])
        })
        
        # Injecter des missing values
        missing_indices_x1 = np.random.choice(n, size=int(n * 0.15), replace=False)
        df.loc[missing_indices_x1, 'x1'] = np.nan
        
        missing_indices_x2 = np.random.choice(n, size=int(n * 0.10), replace=False)
        df.loc[missing_indices_x2, 'x2'] = np.nan
        
        missing_indices_segment = np.random.choice(n, size=int(n * 0.05), replace=False)
        df.loc[missing_indices_segment, 'segment'] = np.nan
        
        # Injecter des doublons
        n_duplicates = int(n * 0.03)
        duplicate_indices = np.random.choice(n, size=n_duplicates, replace=False)
        df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)
        
        # Injecter des outliers extremes dans x1 et x3
        outlier_indices_x1 = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
        df.loc[outlier_indices_x1, 'x1'] = np.random.choice([200, -50], len(outlier_indices_x1))
        
        outlier_indices_x3 = np.random.choice(len(df), size=int(len(df) * 0.015), replace=False)
        df.loc[outlier_indices_x3, 'x3'] = np.random.uniform(100, 200, len(outlier_indices_x3))
        
        # Injecter des valeurs non numeriques dans x2
        type_error_indices = np.random.choice(len(df), size=5, replace=False)
        df.loc[type_error_indices, 'x2'] = 'oops'
        
        return df
    
    def _generate_eda_data(self, n: int) -> pd.DataFrame:
        """
        Genere des donnees pour la phase EDA.
        
        Variables:
        - Numeriques: age, income, spend, visits
        - Categorielles: segment, channel
        - Cible: churn (optionnel)
        
        Defauts: NA legers (5-10%) + outliers (1-2%) sur income
        """
        df = pd.DataFrame({
            'age': np.random.normal(40, 12, n).clip(18, 80).astype(int),
            'income': np.random.lognormal(10.5, 0.5, n),
            'spend': np.random.gamma(2, 500, n),
            'visits': np.random.poisson(5, n),
            'segment': np.random.choice(['A', 'B', 'C'], n, p=[0.4, 0.35, 0.25]),
            'channel': np.random.choice(['web', 'store', 'app'], n, p=[0.45, 0.35, 0.20]),
            'churn': np.random.choice([0, 1], n, p=[0.75, 0.25])
        })
        
        # Injecter des NA legers
        for col in ['age', 'income', 'spend']:
            missing_idx = np.random.choice(n, size=int(n * 0.07), replace=False)
            df.loc[missing_idx, col] = np.nan
        
        # Injecter des outliers dans income
        outlier_idx = np.random.choice(n, size=int(n * 0.015), replace=False)
        df.loc[outlier_idx, 'income'] = np.random.uniform(200000, 500000, len(outlier_idx))
        
        return df
    
    def _generate_mv_data(self, n: int) -> pd.DataFrame:
        """
        Genere des donnees pour la phase analyse multivariee.
        
        - 8 variables numeriques: x1..x8
        - Structure: 3 clusters simules + colinearite
        - NA faibles (2-5%), pas de target
        """
        # Creer 3 clusters
        n_per_cluster = n // 3
        cluster_labels = np.concatenate([
            np.zeros(n_per_cluster),
            np.ones(n_per_cluster),
            np.full(n - 2 * n_per_cluster, 2)
        ])
        np.random.shuffle(cluster_labels)
        
        # Generer des donnees avec structure de clusters
        df = pd.DataFrame()
        
        # Variables principales
        df['x1'] = cluster_labels * 10 + np.random.normal(0, 2, n)
        df['x2'] = (2 - cluster_labels) * 8 + np.random.normal(0, 1.5, n)
        df['x3'] = np.random.normal(50, 10, n)
        df['x4'] = cluster_labels * -5 + np.random.normal(0, 3, n)
        
        # Variables avec colinearite
        df['x5'] = df['x1'] + np.random.normal(0, 1, n)  # Correlee avec x1
        df['x6'] = df['x2'] * 1.5 + np.random.normal(0, 2, n)  # Correlee avec x2
        df['x7'] = np.random.uniform(0, 100, n)  # Independante
        df['x8'] = df['x3'] - df['x4'] + np.random.normal(0, 5, n)  # Combinaison
        
        # Injecter des NA faibles
        for col in df.columns:
            if np.random.random() < 0.6:  # 60% des colonnes ont des NA
                missing_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
                df.loc[missing_idx, col] = np.nan
        
        return df
    
    def _generate_ml_data(self, n: int) -> pd.DataFrame:
        """
        Genere des donnees pour la phase ML (classification binaire).
        
        - Numeriques: x1..x6
        - Categorielle: segment (A/B/C)
        - Cible: target (0/1)
        - Desequilibre controle (70/30) + bruit
        """
        # Generer la cible avec desequilibre
        target = np.random.choice([0, 1], n, p=[0.7, 0.3])
        
        # Generer des features correlees a la cible
        df = pd.DataFrame()
        
        # Features numeriques avec signal
        df['x1'] = target * 5 + np.random.normal(10, 3, n)
        df['x2'] = target * -3 + np.random.normal(20, 4, n)
        df['x3'] = np.random.normal(15, 5, n)  # Bruit
        df['x4'] = target * 2 + np.random.exponential(2, n)
        df['x5'] = np.random.uniform(0, 50, n)  # Bruit
        df['x6'] = target * 4 + np.random.gamma(2, 3, n)
        
        # Feature categorielle
        # Segment A plus probable pour target=1
        segment_probs = np.where(
            target == 1,
            [[0.6, 0.3, 0.1]] * n,
            [[0.2, 0.5, 0.3]] * n
        )
        df['segment'] = [
            np.random.choice(['A', 'B', 'C'], p=prob)
            for prob in segment_probs
        ]
        
        df['target'] = target
        
        # Ajouter un peu de bruit dans les features
        for col in ['x1', 'x2', 'x4', 'x6']:
            noise_idx = np.random.choice(n, size=int(n * 0.1), replace=False)
            df.loc[noise_idx, col] += np.random.normal(0, 10, len(noise_idx))
        
        return df
    
    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        """
        Recupere un dataset depuis le cache.
        
        Args:
            dataset_id: Identifiant du dataset
            
        Returns:
            DataFrame du dataset
            
        Raises:
            KeyError: Si le dataset n'existe pas
        """
        if dataset_id not in self.datasets_cache:
            raise KeyError(f"Dataset {dataset_id} non trouve dans le cache")
        return self.datasets_cache[dataset_id].copy()
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """
        Recupere les informations d'un dataset.
        
        Args:
            dataset_id: Identifiant du dataset
            
        Returns:
            Dictionnaire d'informations
            
        Raises:
            KeyError: Si le dataset n'existe pas
        """
        if dataset_id not in self.datasets_info:
            raise KeyError(f"Dataset {dataset_id} non trouve")
        return self.datasets_info[dataset_id].copy()


# Instance globale du generateur
dataset_generator = DatasetGenerator()
