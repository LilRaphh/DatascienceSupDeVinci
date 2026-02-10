"""
Service ML Baseline (TP4 - phase ml).
Gere l'entrainement, la sauvegarde et la prediction de modeles.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import time
import uuid

from app.schemas.ml import (
    MlTrainParams,
    MlTrainResult,
    MlMetrics,
    MlPredictResult,
    MlPrediction,
    MlModelInfo,
    ModelType
)
from app.utils.storage import storage


class MlService:
    """Service d'apprentissage automatique baseline."""
    
    def train(
        self,
        df: pd.DataFrame,
        params: MlTrainParams
    ) -> Tuple[str, MlTrainResult]:
        """Entraine un modele de classification."""
        start_time = time.time()
        
        # Separer X et y
        if 'target' not in df.columns:
            raise ValueError("Colonne 'target' non trouvee")
        
        y = df['target'].values
        X = df.drop(columns=['target']).copy()
        
        # Encoder les variables categorielles
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        features_used = X.columns.tolist()
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=params.test_size,
            random_state=params.random_state,
            stratify=y
        )
        
        # Entrainer le modele
        if params.model_type == ModelType.LOGREG:
            model = LogisticRegression(
                C=params.logreg_C,
                max_iter=params.logreg_max_iter,
                random_state=params.random_state
            )
        else:  # RF
            model = RandomForestClassifier(
                n_estimators=params.rf_n_estimators,
                max_depth=params.rf_max_depth,
                random_state=params.random_state
            )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Probabilites
        try:
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_train_proba = None
            y_test_proba = None
        
        # Calculer metriques
        metrics_train = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
        metrics_test = self._calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        training_time = time.time() - start_time
        
        # Sauvegarder le modele
        model_id = f"{params.model_type.value}_{uuid.uuid4().hex[:12]}"
        
        model_data = {
            'model': model,
            'label_encoders': label_encoders,
            'features': features_used
        }
        
        metadata = {
            'model_id': model_id,
            'model_type': params.model_type.value,
            'hyperparameters': params.model_dump(),
            'features_used': features_used,
            'preprocessing_steps': ['label_encoding'],
            'performance_metrics': metrics_test.model_dump(),
            'training_time': training_time
        }
        
        storage.save_model(model_id, model_data, metadata)
        
        result = MlTrainResult(
            model_id=model_id,
            model_type=params.model_type.value,
            dataset_id="",
            n_train=len(X_train),
            n_test=len(X_test),
            features_used=features_used,
            metrics_train=metrics_train,
            metrics_test=metrics_test,
            training_time=training_time
        )
        
        return model_id, result
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> MlMetrics:
        """Calcule les metriques de classification."""
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        metrics = MlMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
            auc_roc=float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
            confusion_matrix=cm
        )
        
        return metrics
    
    def predict(
        self,
        model_id: str,
        data: List[Dict[str, Any]]
    ) -> MlPredictResult:
        """Fait des predictions avec un modele."""
        model_data = storage.load_model(model_id)
        
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        features = model_data['features']
        
        # Creer DataFrame
        df = pd.DataFrame(data)
        
        # Encoder les variables categorielles
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        # S'assurer que toutes les features sont presentes
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        
        X = df[features]
        
        # Predictions
        predictions = model.predict(X)
        
        try:
            probabilities = model.predict_proba(X)
        except:
            probabilities = None
        
        results = []
        for i, pred in enumerate(predictions):
            proba = probabilities[i].tolist() if probabilities is not None else None
            results.append(
                MlPrediction(
                    prediction=int(pred),
                    probability=proba
                )
            )
        
        return MlPredictResult(
            model_id=model_id,
            n_predictions=len(results),
            predictions=results
        )
    
    def get_model_info(self, model_id: str) -> MlModelInfo:
        """Recupere les informations d'un modele."""
        metadata = storage.load_metadata(model_id)
        
        return MlModelInfo(**metadata)


# Instance globale
ml_service = MlService()
