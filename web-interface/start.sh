#!/bin/bash

# Script de lancement de l'interface web

echo "🚀 Lancement de l'interface web FastAPI Data Scientist"
echo ""

# Vérifier si l'API tourne
echo "Vérification de l'API..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API en ligne"
else
    echo "⚠️  API hors ligne"
    echo "Lancez d'abord l'API avec : docker-compose up -d"
    echo ""
fi

# Démarrer un serveur HTTP simple
echo ""
echo "Lancement du serveur HTTP sur http://localhost:8088"
echo "Appuyez sur Ctrl+C pour arrêter"
echo ""

python3 -m http.server 8088
