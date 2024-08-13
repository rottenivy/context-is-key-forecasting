import os
import pandas as pd
import requests
from io import StringIO

import numpy as np
import glob

from importlib import resources

from benchmark.config import DATA_STORAGE_PATH

TRAFFIC_STORAGE_PATH = os.environ.get(
    "TRAFFIC_DATA_STORE", os.path.join(DATA_STORAGE_PATH, "traffic_data")
)
TRAFFIC_CSV_PATH = os.path.join(TRAFFIC_STORAGE_PATH, "traffic_merged_sensor_data.csv")
TRAFFIC_SPLIT_PATH = os.path.join(TRAFFIC_STORAGE_PATH, "traffic_split_sensor_data")

traffic_url = "https://huggingface.co/datasets/yatsbm/TrafficFresh/resolve/main/traffic_merged_sensor_data.csv"
hf_token = "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"


def download_traffic_data(hf_token="", traffic_url=traffic_url):
    # Check if the file already exists
    if os.path.exists(TRAFFIC_CSV_PATH):
        print(f"File already exists at {TRAFFIC_CSV_PATH}. Skipping download.")
        return

    # Create storage directory if it doesn't exist
    if not os.path.exists(TRAFFIC_STORAGE_PATH):
        os.makedirs(TRAFFIC_STORAGE_PATH, exist_ok=True)

    # Request the data from Hugging Face with authorization
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.get(traffic_url, headers=headers)

    if response.status_code == 200:
        # Load the CSV data into a pandas DataFrame
        df = pd.read_csv(StringIO(response.text), delimiter="\t", header=0)

        # Perform any desired preprocessing here
        df.to_csv(TRAFFIC_CSV_PATH, header=True, index=False)

        print(f"Data successfully downloaded and saved to {TRAFFIC_CSV_PATH}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")


def split_and_save_wide_dataframes(
    input_path=TRAFFIC_CSV_PATH,
    output_dir=TRAFFIC_SPLIT_PATH,
):
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


if __name__ == "__main__":
    download_traffic_data(
        hf_token,
        traffic_url,
    )
    split_and_save_wide_dataframes()
