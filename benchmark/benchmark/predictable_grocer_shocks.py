import numpy as np

import pandas as pd
import random
import json

from .utils import get_random_window_univar

from .base import UnivariateCRPSTask


GROCER_SALES_INFLUENCES_PATH = "/home/toolkit/starcaster/research-starcaster/benchmark/benchmark/data/grocer_sales_influences.json"
DOMINICK_GROCER_SALES_PATH = "/home/toolkit/starcaster/research-starcaster/benchmark/benchmark/data/filtered_dominic.csv"


class PredictableGrocerPersistentShockUnivariateTask(UnivariateCRPSTask):
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
        self.prediction_length = np.random.randint(7, 30)
        with open(GROCER_SALES_INFLUENCES_PATH, "r") as file:
            self.influences = json.load(file)
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        dataset = pd.read_csv(DOMINICK_GROCER_SALES_PATH)
        dataset["date"] = pd.to_datetime(dataset["date"])
        dataset = dataset.set_index("date")

        sales_categories = ["grocery", "beer", "meat"]
        stores = dataset["store"].unique()

        success_window = False
        counter = 0
        while not success_window:
            # pick a random sales category and store
            sales_category = self.random.choice(sales_categories)
            store = self.random.choice(stores)

            # select a random series
            series = dataset[dataset["store"] == store][sales_category]
            # select a random window
            try:
                history_factor = self.random.randint(3, 7)
                assert len(series) > (history_factor + 1) * self.prediction_length
                window = get_random_window_univar(
                    series,
                    prediction_length=self.prediction_length,
                    history_factor=history_factor,
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
        shock_delay_in_days = self.random.randint(2, self.prediction_length)
        direction = self.random.choice(["positive", "negative"])
        size = self.random.choice(["small", "medium", "large"])
        influence_info = self.influences[sales_category][direction][size]
        impact_range = influence_info["impact"]
        min_magnitude, max_magnitude = map(
            lambda x: int(x.strip("%")), impact_range.split("-")
        )
        impact_magnitude = random.randint(min_magnitude, max_magnitude)

        # apply the influence to the future series
        future_series[shock_delay_in_days:] = self.apply_influence_to_series(
            future_series[shock_delay_in_days:], impact_magnitude, direction
        )

        self.shock_description = influence_info["influence"].replace(
            "{time_in_days}", str(shock_delay_in_days)
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
        self.scenario = self.get_context()

    def apply_influence_to_series(self, series, relative_impact, direction):
        """
        Apply a relative impact to a series
        Parameters:
        -----------
        series: pd.Series
            The series to apply the impact to.
        relative_impact: int
            The relative impact to apply
        direction: str
            The direction of the impact
        Returns:
        --------
        series: pd.Series
            The series with the applied impact
        """
        if direction == "positive":
            series += series * (relative_impact / 100)
        else:
            series -= series * (relative_impact / 100)

        return series

    def get_context(self):
        """
        Get the context of the event.
        Returns:
        --------
        context: str
            The context of the event, including the influence and the relative impact.

        """
        context = self.shock_description
        relative_impact = self.impact_magnitude
        if self.direction == "negative":
            context += f" The relative impact is expected to be a persistent {relative_impact}% decrease."
        else:
            context += f" The relative impact is expected to be a persistent {relative_impact}% increase."
        return context


__TASKS__ = [PredictableGrocerPersistentShockUnivariateTask]
