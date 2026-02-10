"""
Service ML Avance (TP5 - phase ml2).
Gere l'optimisation d'hyperparametres et l'explicabilite.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import uuid

from app.schemas.ml2 import *
from app.services.ml_service import ml_service
from app.utils.storage import storage


class Ml2Service:
    """Service ML avance."""
    
    def tune(
        self,
        df: pd.DataFrame,
        params: Ml2TuneParams
    ) -> Ml2TuneResult:
        """Optimise les hyperparametres avec CV."""
        import time
        start_time = time.time()
        
        # Preparer donnees
        y = df['target'].values
        X = df.drop(columns=['target']).copy()
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Definir param_grid
        if params.model_type == "logreg":
            model = LogisticRegression(max_iter=1000, random_state=42)
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        else:  # rf
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        
        # Recherche
        if params.search == SearchStrategy.GRID:
            search = GridSearchCV(
                model, param_grid, cv=params.cv,
                scoring=params.scoring, n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                model, param_grid, cv=params.cv, n_iter=params.n_iter,
                scoring=params.scoring, random_state=42, n_jobs=-1
            )
        
        search.fit(X_train, y_train)
        
        # Sauvegarder meilleur modele
        best_model_id = f"{params.model_type}_tuned_{uuid.uuid4().hex[:12]}"
        
        model_data = {
            'model': search.best_estimator_,
            'label_encoders': {},
            'features': X.columns.tolist()
        }
        
        y_test_pred = search.best_estimator_.predict(X_test)
        metrics = ml_service._calculate_metrics(y_test, y_test_pred)
        
        storage.save_model(
            best_model_id,
            model_data,
            {
                'model_id': best_model_id,
                'model_type': params.model_type,
                'hyperparameters': search.best_params_,
                'features_used': X.columns.tolist(),
                'performance_metrics': metrics.model_dump()
            }
        )
        
        # Creer resume CV
        cv_results = []
        results_df = pd.DataFrame(search.cv_results_)
        results_df = results_df.sort_values('rank_test_score')
        
        for i, row in results_df.head(5).iterrows():
            cv_results.append(
                CvResult(
                    rank=int(row['rank_test_score']),
                    params=row['params'],
                    mean_score=float(row['mean_test_score']),
                    std_score=float(row['std_test_score']),
                    scores=[float(row[f'split{j}_test_score']) for j in range(params.cv)]
                )
            )
        
        return Ml2TuneResult(
            best_model_id=best_model_id,
            best_params=search.best_params_,
            best_score=float(search.best_score_),
            cv_results_summary=cv_results,
            total_fits=len(search.cv_results_['params']),
            tuning_time=time.time() - start_time
        )
    
    def feature_importance(self, model_id: str) -> Ml2FeatureImportanceResult:
        """Calcule l'importance native des features."""
        model_data = storage.load_model(model_id)
        metadata = storage.load_metadata(model_id)
        
        model = model_data['model']
        features = model_data['features']
        
        # Importance native
        if hasattr(model, 'feature_importances_'):
            importances_values = model.feature_importances_
            method = "native"
        elif hasattr(model, 'coef_'):
            importances_values = np.abs(model.coef_[0])
            method = "coefficients"
        else:
            raise ValueError("Modele ne supporte pas l'importance native")
        
        # Creer liste triee
        sorted_idx = np.argsort(importances_values)[::-1]
        
        importances = []
        for rank, idx in enumerate(sorted_idx, 1):
            importances.append(
                FeatureImportance(
                    feature=features[idx],
                    importance=float(importances_values[idx]),
                    rank=rank
                )
            )
        
        return Ml2FeatureImportanceResult(
            model_id=model_id,
            model_type=metadata['model_type'],
            method=method,
            importances=importances,
            top_features=[imp.feature for imp in importances[:5]]
        )
    
    def permutation_importance_analysis(
        self,
        model_id: str,
        df: pd.DataFrame,
        params: Ml2PermutationParams
    ) -> Ml2PermutationResult:
        """Calcule l'importance par permutation."""
        model_data = storage.load_model(model_id)
        
        model = model_data['model']
        features = model_data['features']
        
        # Preparer donnees
        y = df['target'].values
        X = df.drop(columns=['target'])[features]
        
        # Encoder si necessaire
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Calculer permutation importance
        result = permutation_importance(
            model, X, y,
            n_repeats=params.n_repeats,
            random_state=params.random_state,
            n_jobs=-1
        )
        
        # Trier
        sorted_idx = np.argsort(result.importances_mean)[::-1]
        
        importances = []
        for rank, idx in enumerate(sorted_idx, 1):
            importances.append(
                FeatureImportance(
                    feature=features[idx],
                    importance=float(result.importances_mean[idx]),
                    rank=rank
                )
            )
        
        return Ml2PermutationResult(
            model_id=model_id,
            n_repeats=params.n_repeats,
            importances=importances,
            top_features=[imp.feature for imp in importances[:5]]
        )
    
    def explain_instance(
        self,
        model_id: str,
        instance: Dict[str, Any]
    ) -> Ml2ExplainResult:
        """Explique une prediction."""
        model_data = storage.load_model(model_id)
        metadata = storage.load_metadata(model_id)
        
        model = model_data['model']
        features = model_data['features']
        
        # Preparer instance
        df = pd.DataFrame([instance])
        for col in df.select_dtypes(include=['object']).columns:
            if col in features:
                le = LabelEncoder()
                le.classes_ = np.array(['A', 'B', 'C'])  # Simplification
                try:
                    df[col] = le.transform(df[col])
                except:
                    df[col] = 0
        
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        
        X = df[features]
        
        # Prediction
        pred = int(model.predict(X)[0])
        try:
            proba = float(model.predict_proba(X)[0, 1])
        except:
            proba = None
        
        # Contributions
        contributions = []
        
        if hasattr(model, 'coef_'):
            # LogReg: contribution = coef * valeur
            for i, feat in enumerate(features):
                contrib = float(model.coef_[0, i] * X.iloc[0, i])
                contributions.append(
                    FeatureContribution(
                        feature=feat,
                        value=instance.get(feat, 0),
                        contribution=contrib
                    )
                )
        else:
            # RF: approximation simple
            if hasattr(model, 'feature_importances_'):
                for i, feat in enumerate(features):
                    contrib = float(model.feature_importances_[i] * X.iloc[0, i])
                    contributions.append(
                        FeatureContribution(
                            feature=feat,
                            value=instance.get(feat, 0),
                            contribution=contrib
                        )
                    )
        
        # Trier
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        top_positive = [c.feature for c in contributions if c.contribution > 0][:5]
        top_negative = [c.feature for c in contributions if c.contribution < 0][:5]
        
        summary = f"Prediction: {pred}. "
        if top_positive:
            summary += f"Facteurs positifs: {', '.join(top_positive[:3])}. "
        if top_negative:
            summary += f"Facteurs negatifs: {', '.join(top_negative[:3])}."
        
        return Ml2ExplainResult(
            model_id=model_id,
            prediction=pred,
            probability=proba,
            contributions=contributions,
            top_positive=top_positive,
            top_negative=top_negative,
            explanation_summary=summary
        )


# Instance globale
ml2_service = Ml2Service()
