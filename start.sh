#!/bin/bash

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Configurando el entorno virtual...${NC}"
# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Entorno virtual creado${NC}"
else
    echo -e "${GREEN}Entorno virtual ya existe${NC}"
fi

# Activar entorno virtual
source venv/bin/activate

echo -e "${YELLOW}Instalando dependencias...${NC}"
pip install -r requirements.txt

echo -e "${YELLOW}Iniciando el servidor API...${NC}"
python api_server.py serve