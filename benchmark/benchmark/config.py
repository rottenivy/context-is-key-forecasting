"""
Configuration variables for the benchmark

"""

import os

from pathlib import Path

# Model weight storage
MODEL_STORAGE_PATH = Path(os.environ.get("STARCASTER_MODEL_STORE", "./models"))
if not os.path.exists(MODEL_STORAGE_PATH):
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

# Evaluation configuration
DEFAULT_N_SAMPLES = 50
RESULT_CACHE_PATH = Path(os.environ.get("STARCASTER_RESULT_CACHE", "./inference_cache"))
METRIC_SCALING_CACHE_PATH = os.environ.get(
    "STARCASTER_METRIC_SCALING_CACHE", "./metric_scaling_cache"
)
COMPUTE_METRIC_VARIANCE = bool(
    os.environ.get("STARCASTER_METRIC_COMPUTE_VARIANCE", "")
)  # bool("") == False

# OpenAI configuration
OPENAI_USE_AZURE = (
    os.environ.get("STARCASTER_OPENAI_USE_AZURE", "False").lower() == "true"
)
OPENAI_API_KEY = os.environ.get("STARCASTER_OPENAI_API_KEY", "")
OPENAI_API_VERSION = os.environ.get("STARCASTER_OPENAI_API_VERSION", None)
OPENAI_AZURE_ENDPOINT = os.environ.get("STARCASTER_OPENAI_AZURE_ENDPOINT", None)

# Nixtla configuration
NIXTLA_BASE_URL = os.environ.get("STARCASTER_NIXTLA_BASE_URL", None)
NIXTLA_API_KEY = os.environ.get("STARCASTER_NIXTLA_API_KEY", None)


DATA_STORAGE_PATH = Path(os.environ.get("STARCASTER_DATA_STORE", "benchmark/data"))
if not DATA_STORAGE_PATH.exists():
    DATA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# Data location for the data required for various tasks
DOMINICK_STORAGE_PATH = os.environ.get(
    "STARCASTER_DOMINICK_STORE", os.path.join(DATA_STORAGE_PATH, "dominicks")
)
HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.join(DATA_STORAGE_PATH, "hf_cache"))
TRAFFIC_STORAGE_PATH = os.environ.get(
    "TRAFFIC_DATA_STORE", os.path.join(DATA_STORAGE_PATH, "traffic_data")
)
