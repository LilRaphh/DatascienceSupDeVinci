# Image de base Python
FROM python:3.11-slim

# Definir le repertoire de travail
WORKDIR /code

# Copier le fichier de dependances
COPY requirements.txt .

# Installer les dependances Python
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copier le code de l'application
COPY ./app /code/app

# Creer les repertoires necessaires pour le stockage
RUN mkdir -p /code/models /code/data

# Exposer le port de l'API
EXPOSE 8000

# Commande de demarrage de l'application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
