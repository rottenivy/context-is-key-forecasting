"""
Configuration variables for the benchmark

"""
import os

MODEL_STORAGE_PATH = os.environ.get("STARCASTER_MODEL_STORE", "./models")
if not os.path.exists(MODEL_STORAGE_PATH):
    os.makedirs(MODEL_STORAGE_PATH)