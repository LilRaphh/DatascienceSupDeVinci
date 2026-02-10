"""
Configuration de l'application FastAPI.
Ce fichier centralise tous les parametres de configuration.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Classe de configuration de l'application.
    Les valeurs peuvent etre surchargees par des variables d'environnement.
    """
    
    # Informations de l'API
    app_name: str = "FastAPI - Parcours Data Scientist"
    app_version: str = "1.0.0"
    app_description: str = "API pour les 5 TPs du parcours Data Scientist"
    
    # Configuration de l'environnement
    environment: str = "development"
    log_level: str = "info"
    
    # Chemins de stockage
    models_dir: str = "/code/models"
    data_dir: str = "/code/data"
    
    # Parametres par defaut pour la generation de datasets
    default_n_rows: int = 1000
    default_seed: int = 42
    
    # Configuration CORS (si necessaire)
    cors_origins: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Instance globale de configuration
settings = Settings()
