# unzip and load
import zipfile
import os
import pandas as pd

# download
import urllib.request

from benchmark.config import DOMINICK_STORAGE_PATH

# Dominick's download
# Academic research only
dominicks_url = "https://www.chicagobooth.edu/boothsitecore/docs/dff/store-demos-customer-count/ccount_stata.zip"


def download_dominicks(dominicks_url):

    if not os.path.exists(DOMINICK_STORAGE_PATH):
        os.makedirs(DOMINICK_STORAGE_PATH)

    urllib.request.urlretrieve(
        dominicks_url, os.path.join(DOMINICK_STORAGE_PATH, "dominicks.zip")
    )

    # unzip
    with zipfile.ZipFile("dominicks.zip", "r") as zip_ref:
        zip_ref.extractall("dominicks")

    # load
    filepath = "dominicks/ccount.dta"
    df = pd.read_stata(filepath)
    df = df[(df.week > 0) & (df.week <= 400)]
    df["datetime"] = pd.to_datetime(df["date"], format="%y%m%d", errors="coerce")
    df.set_index("datetime", inplace=True)

    desired_columns = ["store", "grocery", "beer", "meat"]
    df = df[desired_columns].dropna()
    df.dropna(inplace=True).to_csv("filtered_dominick_grocer_sales.csv")


if __name__ == "__main__":
    download_dominicks(dominicks_url)
