"""
Utilitaire de stockage pour les modeles et pipelines.
Gere la serialisation et la deserialisation des objets ML.
"""

import joblib
import json
import os
from typing import Any, Dict
from pathlib import Path


class Storage:
    """
    Classe de gestion du stockage des modeles et pipelines.
    Utilise joblib pour la serialisation des objets scikit-learn.
    """
    
    def __init__(self, base_dir: str = "/code/models"):
        """
        Initialise le gestionnaire de stockage.
        
        Args:
            base_dir: Repertoire de base pour le stockage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache en memoire pour acces rapide
        self._cache: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, Dict] = {}
    
    def save_model(
        self,
        model_id: str,
        model: Any,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Sauvegarde un modele avec ses metadonnees.
        
        Args:
            model_id: Identifiant unique du modele
            model: Objet modele a sauvegarder
            metadata: Dictionnaire de metadonnees
            
        Returns:
            Chemin du fichier sauvegarde
        """
        # Creer le sous-repertoire si necessaire
        model_dir = self.base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modele
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Sauvegarder les metadonnees
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            # Convertir les objets non JSON-serialisables
            serializable_metadata = self._make_serializable(metadata)
            json.dump(serializable_metadata, f, indent=2)
        
        # Mettre en cache
        self._cache[model_id] = model
        self._metadata_cache[model_id] = metadata
        
        return str(model_path)
    
    def load_model(self, model_id: str) -> Any:
        """
        Charge un modele depuis le stockage.
        
        Args:
            model_id: Identifiant du modele
            
        Returns:
            Objet modele charge
            
        Raises:
            FileNotFoundError: Si le modele n'existe pas
        """
        # Verifier le cache
        if model_id in self._cache:
            return self._cache[model_id]
        
        # Charger depuis le disque
        model_path = self.base_dir / model_id / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Modele {model_id} non trouve")
        
        model = joblib.load(model_path)
        
        # Mettre en cache
        self._cache[model_id] = model
        
        return model
    
    def load_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Charge les metadonnees d'un modele.
        
        Args:
            model_id: Identifiant du modele
            
        Returns:
            Dictionnaire de metadonnees
            
        Raises:
            FileNotFoundError: Si les metadonnees n'existent pas
        """
        # Verifier le cache
        if model_id in self._metadata_cache:
            return self._metadata_cache[model_id].copy()
        
        # Charger depuis le disque
        metadata_path = self.base_dir / model_id / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadonnees pour {model_id} non trouvees")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Mettre en cache
        self._metadata_cache[model_id] = metadata
        
        return metadata.copy()
    
    def model_exists(self, model_id: str) -> bool:
        """
        Verifie si un modele existe.
        
        Args:
            model_id: Identifiant du modele
            
        Returns:
            True si le modele existe, False sinon
        """
        model_path = self.base_dir / model_id / "model.joblib"
        return model_path.exists()
    
    def list_models(self) -> list[str]:
        """
        Liste tous les modeles stockes.
        
        Returns:
            Liste des identifiants de modeles
        """
        if not self.base_dir.exists():
            return []
        
        return [
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "model.joblib").exists()
        ]
    
    def delete_model(self, model_id: str) -> bool:
        """
        Supprime un modele du stockage.
        
        Args:
            model_id: Identifiant du modele
            
        Returns:
            True si supprime, False si non trouve
        """
        model_dir = self.base_dir / model_id
        if not model_dir.exists():
            return False
        
        # Supprimer du cache
        self._cache.pop(model_id, None)
        self._metadata_cache.pop(model_id, None)
        
        # Supprimer les fichiers
        import shutil
        shutil.rmtree(model_dir)
        
        return True
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convertit un objet en format JSON-serialisable.
        
        Args:
            obj: Objet a convertir
            
        Returns:
            Objet serialisable
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)


# Instance globale de stockage
storage = Storage()
