"""
Service ML Baseline (TP4 - phase ml).
Gere l'entrainement, la sauvegarde et la prediction de modeles.
"""

import time
import uuid
from typing import Any, Dict, List, Tuple
from sklearn.impute import SimpleImputer


import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.schemas.ml import (
    MlMetrics,
    MlModelInfo,
    MlPredictResult,
    MlPrediction,
    MlTrainParams,
    MlTrainResult,
    ModelType,
)
from app.utils.storage import storage


class MlService:
    """Service d'apprentissage automatique baseline."""

    def train(
    self,
    dataset_id: str,
    df: pd.DataFrame,
    params: MlTrainParams,
) -> Tuple[str, MlTrainResult]:
        """Entraine un modele de classification."""
        start_time = time.time()

        # Validation cible
        if "target" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Dataset {dataset_id} incompatible: colonne 'target' absente. "
                    f"Utilise un dataset de phase 'ml'/'ml2'. Colonnes: {list(df.columns)}"
                ),
            )

        # Séparer X et y
        y = df["target"].values
        X = df.drop(columns=["target"]).copy()

        # --- Imputation des valeurs manquantes (NaN) ---
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        imp_num = None
        imp_cat = None

        if len(num_cols) > 0:
            imp_num = SimpleImputer(strategy="median")
            X[num_cols] = imp_num.fit_transform(X[num_cols])

        if len(cat_cols) > 0:
            imp_cat = SimpleImputer(strategy="most_frequent")
            X[cat_cols] = imp_cat.fit_transform(X[cat_cols])

        # Encoder les variables categorielles
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        label_encoders: Dict[str, LabelEncoder] = {}

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        features_used = X.columns.tolist()

        # Split train/test (fallback si stratify impossible)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=params.test_size,
                random_state=params.random_state,
                stratify=y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=params.test_size,
                random_state=params.random_state,
                stratify=None,
            )

        # Entrainer le modele
        if params.model_type == ModelType.LOGREG:
            model = LogisticRegression(
                C=params.logreg_C,
                max_iter=params.logreg_max_iter,
                random_state=params.random_state,
            )
        else:  # RF
            model = RandomForestClassifier(
                n_estimators=params.rf_n_estimators,
                max_depth=params.rf_max_depth,
                random_state=params.random_state,
            )

        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilites (binaire uniquement)
        y_train_proba = None
        y_test_proba = None
        try:
            if hasattr(model, "predict_proba") and len(getattr(model, "classes_", [])) == 2:
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_train_proba = None
            y_test_proba = None

        # Metriques
        metrics_train = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
        metrics_test = self._calculate_metrics(y_test, y_test_pred, y_test_proba)

        training_time = time.time() - start_time

        # Sauvegarder le modele
        model_id = f"{params.model_type.value}_{uuid.uuid4().hex[:12]}"

        model_data = {
            "model": model,
            "label_encoders": label_encoders,
            "imputers": {  # pour predict() (cohérence preprocessing)
                "num": imp_num,
                "cat": imp_cat,
            },
            "features": features_used,
        }

        metadata = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "model_type": params.model_type.value,
            "hyperparameters": params.model_dump(),
            "features_used": features_used,
            "preprocessing_steps": ["imputation_median", "imputation_most_frequent", "label_encoding"],
            "performance_metrics": metrics_test.model_dump(),
            "training_time": training_time,
        }

        storage.save_model(model_id, model_data, metadata)

        result = MlTrainResult(
            model_id=model_id,
            model_type=params.model_type.value,
            dataset_id=dataset_id,
            n_train=len(X_train),
            n_test=len(X_test),
            features_used=features_used,
            metrics_train=metrics_train,
            metrics_test=metrics_test,
            training_time=training_time,
        )

        return model_id, result


    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> MlMetrics:
        """Calcule les metriques de classification."""
        cm = confusion_matrix(y_true, y_pred).tolist()

        return MlMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
            auc_roc=float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
            confusion_matrix=cm,
        )

    def predict(self, model_id: str, data: List[Dict[str, Any]]) -> MlPredictResult:
        """Fait des predictions avec un modele."""
        model_data = storage.load_model(model_id)

        model = model_data["model"]
        label_encoders: Dict[str, LabelEncoder] = model_data["label_encoders"]
        features: List[str] = model_data["features"]

        df = pd.DataFrame(data)

        # Encoder les variables categorielles
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)

                def _encode(x: str) -> int:
                    return int(le.transform([x])[0]) if x in le.classes_ else -1

                df[col] = df[col].apply(_encode)

        # S'assurer que toutes les features sont presentes
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0

        X = df[features]

        predictions = model.predict(X)

        probabilities = None
        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)
        except Exception:
            probabilities = None

        results: List[MlPrediction] = []
        for i, pred in enumerate(predictions):
            proba = probabilities[i].tolist() if probabilities is not None else None
            results.append(MlPrediction(prediction=int(pred), probability=proba))

        return MlPredictResult(
            model_id=model_id,
            n_predictions=len(results),
            predictions=results,
        )

    def get_model_info(self, model_id: str) -> MlModelInfo:
        """Recupere les informations d'un modele."""
        metadata = storage.load_metadata(model_id)
        return MlModelInfo(**metadata)


ml_service = MlService()
