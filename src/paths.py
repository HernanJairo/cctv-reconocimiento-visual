"""Resolución centralizada de rutas absolutas del proyecto.

Todos los módulos resuelven sus rutas a partir de BASE_DIR para que la
aplicación funcione sin importar el directorio de trabajo desde el que
se ejecute.
"""
import os

# Raíz del proyecto (carpeta que contiene a src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")
CAPTURAS_DIR = os.path.join(DATA_DIR, "capturas")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EXPORTS_DIR = os.path.join(DATA_DIR, "reportes")

SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.yaml")
DB_PATH = os.path.join(DATA_DIR, "seguridad.db")

for _d in (CONFIG_DIR, DATA_DIR, CAPTURAS_DIR, VIDEOS_DIR, MODELS_DIR, EXPORTS_DIR):
    os.makedirs(_d, exist_ok=True)
