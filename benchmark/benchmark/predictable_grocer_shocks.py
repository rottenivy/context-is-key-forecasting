import numpy as np

import pandas as pd
import random
import json

from .utils import get_random_window_univar

from .base import BaseTask


GROCER_SALES_INFLUENCES_PATH = "/home/toolkit/starcaster/research-starcaster/benchmark/benchmark/data/grocer_sales_influences.json"
DOMINICK_GROCER_SALES_PATH = "/home/toolkit/starcaster/research-starcaster/benchmark/benchmark/data/filtered_dominic.csv"


class PredictableGrocerSpikesUnivariateTask(BaseTask):
    """
    A task where the time series contains spikes that are predictable based on the
    contextual information provided with the data. The spikes should be reflected in
    the forecast.
    Note: this does NOT use the Monash dominick's dataset, which is transformed with no
    meaningful context.
    Parameters:
    -----------
    fixed_config: dict
        A dictionary containing fixed parameters for the task
    seed: int
        Seed for the random number generator
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        super().__init__(seed=seed, fixed_config=fixed_config)

        self.prediction_length = np.random.randint(7, 30)
        with open(GROCER_SALES_INFLUENCES_PATH, "r") as file:
            self.influences = json.load(file)

    def random_instance(self):
        dataset = pd.read_csv(DOMINICK_GROCER_SALES_PATH)
        dataset["date"] = pd.to_datetime(dataset["date"])
        dataset = dataset.set_index("date")

        sales_categories = ["grocery", "beer", "meat"]
        stores = dataset["store"].unique()

        # pick a random sales category and store
        sales_category = self.random.choice(sales_categories)
        store = self.random.choice(stores)

        # select a random series
        series = dataset[dataset["store"] == store][sales_category]
        # select a random window, keep trying until successful
        success_window = False
        counter = 0
        while not success_window and counter < 100:
            try:
                window = get_random_window_univar(
                    series,
                    prediction_length=self.prediction_length,
                    history_factor=self.random.randint(3, 7),
                    random=self.random,
                )
                success_window = True
            except:
                counter += 1
                raise ValueError("Could not find a valid window")

        # extract the history and future series
        history_series = window.iloc[: -self.prediction_length]
        future_series = window.iloc[-self.prediction_length :]
        ground_truth = future_series.copy()

        # choose an influence and a relative impact from the influence
        time_in_days = self.random.randint(2, self.prediction_length)
        direction = self.random.choice(["positive", "negative"])
        size = self.random.choice(["small", "medium", "large"])
        influence_info = self.influences[sales_category][direction][size]
        impact_range = influence_info["impact"]
        min_magnitude, max_magnitude = map(
            lambda x: int(x.strip("%")), impact_range.split("-")
        )
        impact_magnitude = random.randint(min_magnitude, max_magnitude)

        # apply the influence to the future series
        future_series[time_in_days:] = self.apply_influence_to_series(
            future_series[time_in_days:], impact_magnitude, direction
        )

        self.influence = influence_info["influence"].replace(
            "{time_in_days}", str(time_in_days)
        )
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.impact_magnitude = impact_magnitude
        self.direction = direction
        self.ground_truth = ground_truth

        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = None
        self.scenario = self.get_context_from_event()

    def apply_influence_to_series(self, series, relative_impact, direction):
        if direction == "positive":
            series += series * (relative_impact / 100)
        else:
            series -= series * (relative_impact / 100)

        return series

    def get_context_from_event(self):
        context = self.influence
        relative_impact = self.impact_magnitude
        if self.direction == "negative":
            relative_impact = -self.impact_magnitude
        context += f" The relative impact is expected to be {relative_impact}%."
        return context

    def evaluate(self, samples):
        """
        Evaluate the forecast for the grocer spike task.
        Calculate the mean relative change of the forecast vs the ground truth,
        and compare to the relative impact of the influence.
        """
        # Calculate the relative change of the forecast vs the ground truth
        if len(samples.shape) == 3:
            samples = samples[:, :, 0]
        mean_forecast_change = np.mean(np.array(samples) - np.array(self.ground_truth))
        relative_change = mean_forecast_change / np.mean(self.ground_truth + 1e-6)

        # Calculate the relative impact of the influence
        relative_impact = self.impact_magnitude / 100

        if self.direction == "negative":
            relative_impact = -relative_impact

        # Calculate the difference between the relative change and the relative impact
        difference = np.abs(relative_change - relative_impact)

        # Calculate the mean difference across the prediction length
        metric = np.mean(difference)

        return metric
