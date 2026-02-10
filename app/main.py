"""
Application principale FastAPI - Parcours Data Scientist.
Point d'entree de l'API regroupant tous les routers des 5 TPs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import dataset, clean, eda, mv, ml, ml2


# Creation de l'application FastAPI
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routers
app.include_router(
    dataset.router,
    prefix="/dataset",
    tags=["Dataset Generation"]
)

app.include_router(
    clean.router,
    prefix="/clean",
    tags=["TP1 - Nettoyage & Preparation"]
)

app.include_router(
    eda.router,
    prefix="/eda",
    tags=["TP2 - EDA (Exploratory Data Analysis)"]
)

app.include_router(
    mv.router,
    prefix="/mv",
    tags=["TP3 - Analyse Multivariee (PCA & Clustering)"]
)

app.include_router(
    ml.router,
    prefix="/ml",
    tags=["TP4 - ML Baseline"]
)

app.include_router(
    ml2.router,
    prefix="/ml2",
    tags=["TP5 - ML Avance (Tuning & Explicabilite)"]
)


@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint racine de l'API.
    Renvoie les informations de base sur l'API.
    """
    return {
        "message": "Bienvenue sur l'API FastAPI - Parcours Data Scientist",
        "version": settings.app_version,
        "documentation": "/docs",
        "phases_disponibles": [
            "clean - TP1: Nettoyage & Preparation",
            "eda - TP2: EDA (Exploratory Data Analysis)",
            "mv - TP3: Analyse Multivariee (PCA & Clustering)",
            "ml - TP4: ML Baseline",
            "ml2 - TP5: ML Avance (Tuning & Explicabilite)"
        ],
        "endpoints_principaux": {
            "generation_dataset": "POST /dataset/generate",
            "nettoyage": "POST /clean/*",
            "eda": "POST /eda/*",
            "multivarié": "POST /mv/*",
            "ml_baseline": "POST /ml/*",
            "ml_avance": "POST /ml2/*"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Endpoint de verification de sante de l'API.
    Utile pour les checks de monitoring.
    """
    return {
        "status": "healthy",
        "environment": settings.environment
    }
