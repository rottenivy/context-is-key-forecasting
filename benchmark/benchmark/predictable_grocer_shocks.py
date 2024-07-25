import numpy as np

import pandas as pd
import random
import json

from .utils import get_random_window_univar

from .base import UnivariateCRPSTask


class PredictableGrocerPersistentShockUnivariateTask(UnivariateCRPSTask):
    """
    A task where the time series contains spikes that are predictable based on the
    contextual information provided with the data. The spikes should be reflected in
    the forecast.
    Note: this does NOT use the Monash dominick's dataset, which is transformed with no
    meaningful context.
    Context: synthetic
    Series: modified
    Dataset: Dominick's grocer dataset.
    Parameters:
    -----------
    fixed_config: dict
        A dictionary containing fixed parameters for the task
    seed: int
        Seed for the random number generator
    GROCER_SALES_INFLUENCES_PATH: str
        Path to the JSON file containing the sales influences.
    DOMINICK_GROCER_SALES_PATH: str
        Path to the filtered Dominick's grocer dataset.
        Filtered for a subset of products for which we generated influences.
    """

    def __init__(
        self,
        fixed_config: dict = None,
        seed: int = None,
        grocer_sales_influences_path=(
            "/starcaster/data/benchmark/grocer/grocer_sales_influences.json"
        ),
        dominick_grocer_sales_path="/starcaster/data/benchmark/grocer/filtered_dominic.csv",
    ):
        self.dominick_grocer_sales_path = dominick_grocer_sales_path
        self.prediction_length = np.random.randint(7, 30)
        with open(grocer_sales_influences_path, "r") as file:
            self.influences = json.load(file)
        super().__init__(seed=seed, fixed_config=fixed_config)

    def random_instance(self):
        dataset = pd.read_csv(self.dominick_grocer_sales_path)
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
        self.min_magnitude, self.max_magnitude = map(
            lambda x: int(x.strip("%")), impact_range.split("-")
        )
        impact_magnitude = random.randint(self.min_magnitude, self.max_magnitude)

        # apply the influence to the future series
        future_series[shock_delay_in_days:] = self.apply_influence_to_series(
            future_series[shock_delay_in_days:], impact_magnitude, direction
        )

        self.min_magnitude = self.min_magnitude
        self.max_magnitude = self.max_magnitude
        self.impact_magnitude = impact_magnitude
        self.direction = direction
        self.ground_truth = ground_truth

        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = None
        self.scenario = self.get_scenario_context(shock_delay_in_days, influence_info)

    def get_shock_description(self, shock_delay_in_days, influence_info):
        return influence_info["influence"].replace(
            "{time_in_days}", str(shock_delay_in_days)
        )

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

    def get_scenario_context(self, shock_delay_in_days, influence_info):
        """
        Get the context of the event.
        Returns:
        --------
        context: str
            The context of the event, including the influence and the relative impact.

        """
        relative_impact = self.impact_magnitude
        if self.direction == "negative":
            relative_impact = self.impact_magnitude * -1

        shock_description = influence_info["influence"].replace(
            "{time_in_days}", str(shock_delay_in_days)
        )
        shock_description = shock_description.replace(
            "{impact}", str(relative_impact) + "%"
        )
        return shock_description


__TASKS__ = [PredictableGrocerPersistentShockUnivariateTask]
