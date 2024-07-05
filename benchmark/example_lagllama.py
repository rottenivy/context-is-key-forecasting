from itertools import islice

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

from lag_llama.gluon.estimator import LagLlamaEstimator


def get_lag_llama_predictions(
    dataset,
    prediction_length,
    device,
    batch_size=1,
    context_length=32,
    use_rope_scaling=False,
    num_samples=100,
    num_parallel_samples=100,
):
    """
    Generates forecasts using the Lag-Llama model.

    Parameters:
    dataset (Dataset): The dataset to generate predictions for.
    prediction_length (int): The number of timesteps to predict.
    device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
    batch_size (int, optional): The batch size to use. Default is 1.
    context_length (int, optional): The context length for the model. Default is 32.
    use_rope_scaling (bool, optional): Whether to use ROPE scaling. Default is False.
    num_samples (int, optional): The number of samples to generate for each timestep. Default is 100.
    num_parallel_samples (int, optional): The number of parallel samples to generate for each timestep. This should be equal to or less than num_samples. Default is 100.

    Returns:
    tuple: A tuple containing:
        - forecasts (list): A list of forecast objects. Each forecast is of shape (num_samples, prediction_length).
        - tss (list): A list of time series objects with the ground truth corresponding to the forecasts. Each time series is of shape (prediction length,).
    """

    ckpt = torch.load(
        "/Users/alexandre.drouin/.cache/huggingface/hub/models--time-series-foundation-models--Lag-Llama/snapshots/72dcfc29da106acfe38250a60f4ae29d1e56a3d9/lag-llama.ckpt",
        map_location=device,
    )  # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(
            1.0, (context_length + prediction_length) / estimator_args["context_length"]
        ),
    }

    estimator = LagLlamaEstimator(
        ckpt_path="/Users/alexandre.drouin/.cache/huggingface/hub/models--time-series-foundation-models--Lag-Llama/snapshots/72dcfc29da106acfe38250a60f4ae29d1e56a3d9/lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,  # Lag-Llama was trained with a context length of 32, but can work with any context length
        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,
        batch_size=batch_size,
        num_parallel_samples=num_parallel_samples,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss


def prepare_dataset(history, forecast):
    series = pd.concat((history, forecast))
    df = series.to_frame(name="target")
    ds = PandasDataset({"F": df}, target="target")
    return ds


if __name__ == "__main__":
    from benchmark.misleading_history import (
        SensorPeriodicMaintenanceTask,
        SensorTrendAccumulationTask,
    )

    for i in range(20):
        task = SensorTrendAccumulationTask()

        dataset = prepare_dataset(task.past_time, task.future_time)
        forecasts, gt = get_lag_llama_predictions(
            dataset, len(task.future_time), torch.device("mps")
        )
        print(task.evaluate(forecasts[0].samples))

        plt.figure(figsize=(20, 15))
        date_formater = mdates.DateFormatter("%b, %d")
        plt.rcParams.update({"font.size": 15})

        # Iterate through the first 9 series, and plot the predicted samples
        for idx, (forecast, ts) in islice(enumerate(zip(forecasts, gt)), 9):
            ax = plt.subplot(3, 3, idx + 1)

            plt.plot(
                ts[-4 * len(task.future_time) :].to_timestamp(),
                label="target",
            )
            forecast.plot(color="g")
            plt.xticks(rotation=60)
            ax.xaxis.set_major_formatter(date_formater)
            ax.set_title(forecast.item_id)

        plt.gcf().tight_layout()
        plt.legend()
        plt.savefig(f"lag_llama_{task.__class__.__name__}_{i}.png", bbox_inches="tight")
