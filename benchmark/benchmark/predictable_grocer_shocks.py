import numpy as np

import pandas as pd
import random
import json

from .utils import get_random_window_univar

from .base import BaseTask


class SalesInfluences:
    """
    Class to load and access sales influences from a JSON file.
    These influences are used to generate predictable shocks in the sales data.
    """

    def __init__(self, json_path):
        with open(json_path, "r") as file:
            self.influences = json.load(file)

    def get_influence(self, category, direction, size, time_in_days):
        influence_info = self.influences[category][direction][size]
        influence = influence_info["influence"].replace(
            "{time_in_days}", str(time_in_days)
        )
        impact_range = influence_info["impact"]
        min_impact, max_impact = map(
            lambda x: int(x.strip("%")), impact_range.split("-")
        )
        random_impact = random.randint(min_impact, max_impact)

        return influence, random_impact


class PredictableGrocerSpikesUnivariateTask(BaseTask):
    """
    A task where the time series contains spikes that are predictable based on the
    contextual information provided with the data. The spikes should be reflected in
    the forecast.
    Note: this does NOT use the Monash dominick's dataset, which is transformed with no
    meaningful context.
    Time series: modified from real
    Context: synthetic
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):
        self.prediction_length = np.random.randint(7, 30)
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        dataset = self.load_dataset("dominick")
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
        while not success_window and counter < 10:
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

        # choose an influence
        time_in_days = self.random.randint(2, self.prediction_length)
        direction = self.random.choice(["positive", "negative"])
        size = self.random.choice(["small", "medium", "large"])
        influence, relative_impact = self.get_influence(
            sales_category, direction, size, time_in_days
        )
        self.relative_impact = relative_impact
        self.direction = direction
        self.ground_truth = future_series.copy()

        future_series[time_in_days:] = self.add_influence(
            future_series[time_in_days:], relative_impact, direction
        )

        self.past_time = history_series
        self.future_time = future_series
        self.constraints = None
        self.background = influence
        self.scenario = None
        self.past_context = None
        self.future_context = None

    def get_influence(self, category, direction, size, time_in_days):
        influences = SalesInfluences(
            "/home/toolkit/starcaster/research-starcaster/benchmark/benchmark/data/grocer_sales_influences.json"
        )
        influence, random_impact = influences.get_influence(
            category, direction, size, time_in_days
        )

        return influence, random_impact

    def add_influence(self, series, relative_impact, direction):
        if direction == "positive":
            series += series * (relative_impact / 100)
        else:
            series -= series * (relative_impact / 100)

        return series

    def load_dataset(self, dataset_name):
        if dataset_name == "dominick":
            dataset = pd.read_csv(
                "/home/toolkit/starcaster/research-starcaster/benchmark/benchmark/data/filtered_dominic.csv"
            )

        return dataset

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
        relative_change = mean_forecast_change / np.mean(self.ground_truth)

        # Calculate the relative impact of the influence
        relative_impact = self.relative_impact / 100

        if self.direction == "negative":
            relative_impact = -relative_impact

        # Calculate the difference between the relative change and the relative impact
        difference = np.abs(relative_change - relative_impact)

        # Calculate the mean difference across the prediction length
        metric = np.mean(difference)

        return metric
