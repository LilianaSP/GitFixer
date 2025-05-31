import os
import requests

PROJECT_ROOT = "/Users/lisolorz/Desktop/huggingface"
API_URL = "http://localhost:7774/chat"
README_PATH = os.path.join(PROJECT_ROOT, "README.md")

# Recorrer archivos y carpetas
file_tree = []
for root, dirs, files in os.walk(PROJECT_ROOT):
    # Ignorar carpetas ocultas y venv
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv' and d != '__pycache__']
    rel_root = os.path.relpath(root, PROJECT_ROOT)
    for d in dirs:
        file_tree.append(os.path.join(rel_root, d) + "/")
    for f in files:
        file_tree.append(os.path.join(rel_root, f))

file_tree_str = '\n'.join(sorted(file_tree))

PROMPT = f'''
Rol: Eres un asistente de IA experto en desarrollo de software y documentación técnica.

Tarea:
Analiza exhaustivamente todos los archivos y subdirectorios dentro de la carpeta actual del proyecto. Con base en este análisis, genera un archivo README.md completo y bien estructurado en formato Markdown. El objetivo de este README.md es proporcionar un conocimiento integral del proyecto, explicar cómo funciona y detallar cómo un nuevo usuario puede configurarlo y ejecutarlo.

Contexto Importante:
El proyecto utiliza un ambiente virtual de Python para la gestión de sus dependencias.

A continuación, la estructura de archivos y carpetas del proyecto:
{file_tree_str}

Por favor, incluye las siguientes secciones en el README.md:

[Sugerir Título del Proyecto Aquí]
(IA: Basándote en el contenido y los nombres de los archivos, sugiere un título apropiado para el proyecto. Si no es obvio, usa un placeholder como "[Nombre del Proyecto]".)

1. 📜 Descripción General del Proyecto
Describe brevemente el propósito principal del proyecto. ¿Qué problema resuelve o qué tarea realiza?
Menciona las funcionalidades clave o los objetivos que busca alcanzar.
2. 📋 Prerrequisitos
Especifica la versión de Python recomendada o mínima para ejecutar el proyecto (ej. Python 3.8+).
Lista cualquier otra dependencia a nivel de sistema o herramienta externa que sea necesaria antes de la configuración (ej. Git, Docker, una base de datos específica, etc.).
3. 📁 Estructura del Proyecto
Proporciona un resumen de la organización de los archivos y directorios más importantes.
Explica brevemente el rol de cada componente principal (ej. src/ para el código fuente, data/ para archivos de datos, tests/ para pruebas, main.py como script de entrada, utils.py para funciones de utilidad, etc.).
4. 💡 Cómo Funciona
Explica la arquitectura general o el flujo de trabajo del proyecto a un nivel conceptual.
¿Cómo interactúan los diferentes módulos, scripts o clases principales?
Si hay algún algoritmo o lógica central importante, descríbelo brevemente.
5. ⚙️ Configuración e Instalación
Proporciona instrucciones detalladas paso a paso para configurar el entorno de desarrollo:

Obtención del código (ej. "Clona este repositorio" o "Asegúrate de tener todos los archivos del proyecto en tu máquina local").
Creación y activación del ambiente virtual de Python:
Instruye cómo crear el ambiente (ej. python3 -m venv .venv o python -m venv venv).
Instruye cómo activarlo en diferentes sistemas operativos (ej. source .venv/bin/activate para macOS/Linux y .venv\Scripts\activate para Windows).
Instalación de dependencias:
Si existe un archivo requirements.txt, indica que se use pip install -r requirements.txt dentro del ambiente activado.
Si no existe requirements.txt, analiza las importaciones en los archivos Python del proyecto para identificar las bibliotecas externas principales. Lista los comandos pip install <biblioteca> necesarios. Adicionalmente, sugiere al usuario crear un archivo requirements.txt ejecutando pip freeze > requirements.txt después de instalar las dependencias.
6. ▶️ Cómo Ejecutar el Proyecto
Indica claramente el o los comandos exactos para iniciar o ejecutar la aplicación o el script principal del proyecto (ej. python main.py, python -m src.app).
Menciona cualquier argumento de línea de comandos necesario u opcional.
Si el proyecto requiere la configuración de variables de entorno o archivos de configuración (ej. un .env o config.ini), explica cómo y dónde configurarlos con ejemplos si es posible.
Si hay diferentes modos de ejecución (ej. desarrollo, producción, con/sin ciertas características), explica cómo acceder a ellos.
7. ✨ Ejemplo de Uso (Opcional pero Recomendado)
(IA: Si es posible inferir un caso de uso simple a partir del código, proporciona un pequeño ejemplo de cómo se podría utilizar el proyecto o qué tipo de input espera y qué output produce).

Formato de Salida:
El resultado final debe ser el contenido completo del archivo README.md en formato Markdown. Asegúrate de que la redacción sea clara, concisa y fácil de seguir para alguien que es nuevo en el proyecto.
'''

# Llamar al endpoint local
response = requests.post(API_URL, json={"prompt": PROMPT})
if response.status_code == 200:
    content = response.json().get("response", "")
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"README.md generado en {README_PATH}")
else:
    print(f"Error al generar README.md: {response.text}")