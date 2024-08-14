import os
import pandas as pd
import requests
from io import StringIO
import json

import numpy as np
import glob

from importlib import resources

import huggingface_hub
import datasets


from benchmark.config import DATA_STORAGE_PATH

HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.join(DATA_STORAGE_PATH, "hf_cache"))

TRAFFIC_STORAGE_PATH = os.environ.get(
    "TRAFFIC_DATA_STORE", os.path.join(DATA_STORAGE_PATH, "traffic_data")
)
TRAFFIC_CSV_PATH = os.path.join(TRAFFIC_STORAGE_PATH, "traffic_merged_sensor_data.csv")
TRAFFIC_METADATA_PATH = os.path.join(TRAFFIC_STORAGE_PATH, "traffic_metadata.json")
TRAFFIC_SPLIT_PATH = os.path.join(TRAFFIC_STORAGE_PATH, "traffic_split_sensor_data")

traffic_url = "https://huggingface.co/datasets/yatsbm/TrafficFresh/resolve/main/traffic_merged_sensor_data.csv"
hf_token = ""

# avoid issues where toolkit does not report memory correctly
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

csv_file_path = huggingface_hub.hf_hub_download(
    repo_id="yatsbm/TrafficFresh",
    filename="traffic_merged_sensor_data.csv",
    repo_type="dataset",
    cache_dir=HF_CACHE_DIR,
    token=True,
)

json_file_path = huggingface_hub.hf_hub_download(
    repo_id="yatsbm/TrafficFresh",
    filename="traffic_metadata.json",
    repo_type="dataset",
    cache_dir=HF_CACHE_DIR,
    token=True,
)

# Print the file paths
print(f"CSV File Path: {csv_file_path}")
print(f"JSON File Path: {json_file_path}")


def download_raw_traffic_data():
    # Check if the file already exists
    # if os.path.exists(TRAFFIC_CSV_PATH):
    #     print(f"File already exists at {TRAFFIC_CSV_PATH}. Skipping download.")
    #     return

    # Create storage directory if it doesn't exist
    if not os.path.exists(TRAFFIC_STORAGE_PATH):
        os.makedirs(TRAFFIC_STORAGE_PATH, exist_ok=True)

    csv_file_path = huggingface_hub.hf_hub_download(
        repo_id="yatsbm/TrafficFresh",
        filename="traffic_merged_sensor_data.csv",
        repo_type="dataset",
        cache_dir=HF_CACHE_DIR,
    )

    df = pd.read_csv(csv_file_path, delimiter="\t", header=0)

    # Perform any desired preprocessing here
    df.to_csv(TRAFFIC_CSV_PATH, header=True, index=False)

    print(f"Data downloaded and saved to {TRAFFIC_CSV_PATH}")


def download_traffic_metadata():

    json_file_path = huggingface_hub.hf_hub_download(
        repo_id="yatsbm/TrafficFresh",
        filename="traffic_metadata.json",
        repo_type="dataset",
        cache_dir=HF_CACHE_DIR,
    )
    with open(json_file_path, "r") as f:
        metadata = json.load(f)
        # Save the metadata to a JSON file
        with open(TRAFFIC_METADATA_PATH, "w") as f:
            json.dump(metadata, f)


def split_and_save_wide_dataframes(
    input_path=TRAFFIC_CSV_PATH,
    output_dir=TRAFFIC_SPLIT_PATH,
):
    if os.path.exists(output_dir):
        print(f"Data already split and saved to {output_dir}. Skipping split.")
        return
    # Load the combined data
    combined_data = pd.read_csv(input_path, header=0)

    # Create a directory to save the dataframes if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each unique sensor_id
    for sensor_id in combined_data["sensor_id"].unique():
        # Filter the DataFrame for the current sensor_id
        sensor_df = combined_data[combined_data["sensor_id"] == sensor_id]

        # sort the DataFrame by the datetime
        if not sensor_df["Hour"].is_monotonic_increasing:
            sensor_df = sensor_df.sort_values("Hour")

        # Define the output file name based on sensor_id
        output_file = os.path.join(output_dir, f"sensor_{sensor_id}.csv")

        # Save the wide DataFrame to a CSV file
        sensor_df.to_csv(output_file, index=False)


def download_traffic_files():
    download_raw_traffic_data()
    download_traffic_metadata()
    split_and_save_wide_dataframes()


if __name__ == "__main__":
    download_traffic_files()
