import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path


logging.basicConfig(level=logging.INFO)


METRIC_NAMES = {
    "standard_crps": "unweighted_CRPS",
    "crps": "RCRPS",
    "roi_crps": "RoI_only_CRPS",
    "non_roi_crps": "nonRoI_only_CRPS",
}


def get_df(input_folder) -> pd.DataFrame:
    entries = []
    for f in input_folder.glob("*/*/*/*/evaluation"):
        instance = f.parts[-2]
        task = f.parts[-3]
        model = f.parts[-4]
        model_family = f.parts[-5]

        s = open(f, "r").read()
        try:
            entry = eval(s)

            entry["model_family"] = model_family
            entry["model"] = model
            entry["task"] = task
            entry["instance"] = instance

            entries.append(entry)
        except:
            logging.info(f"Cannot read file for: {model}, {task}, {instance}")
    return pd.DataFrame(entries)


def get_pivot_table(df: pd.DataFrame, metric) -> pd.DataFrame:
    def aggfunc(x):
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        stderr = std / np.sqrt(len(x))
        return f"{mean:.3f} ± {stderr:.3f}"

    return pd.pivot_table(
        df, values=metric, index=["task"], columns=["model_family"], aggfunc=aggfunc
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/starcaster/data/benchmark/results",
        help="Input folder containing experiment results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/starcaster/data/benchmark/results",
        help="Output folder for results",
    )

    args = parser.parse_args()
    input_folder = Path(args.input)
    output_folder = Path(args.output)

    df = get_df(input_folder)
    # Add the violation_crps to the various CRPS versions
    for metric in METRIC_NAMES.keys():
        df[metric] = df[metric] + df.violation_crps
    # Only keep tasks with non-trivial RoI definition
    roi_df = df[(df.num_non_roi_timesteps > 0) & (df.num_roi_timesteps > 0)]
    for metric, output_name in METRIC_NAMES.items():
        pivot_df = get_pivot_table(roi_df, metric)
        pivot_df = pivot_df.sort_index()
        pivot_df.to_csv(output_folder / f"results_{output_name}.csv")


if __name__ == "__main__":
    main()
