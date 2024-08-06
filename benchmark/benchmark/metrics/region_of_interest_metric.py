import numpy as np
import matplotlib.pyplot as plt

from crps import crps_quantile
import numpy as np


class BaseRegionOfInterestMetric:
    def __init__(self, metric):
        self.metric = metric

    def _process_int_region(self, region_of_interest, forecast_shape):
        mask = np.zeros(forecast_shape[1], dtype=bool)
        mask[region_of_interest] = True
        return mask

    def _process_list_region(self, region_of_interest, forecast_shape):
        mask = np.zeros(forecast_shape[1], dtype=bool)
        for i in region_of_interest:
            mask[i] = True
        return mask

    def _process_slice_region(self, region_of_interest, forecast_shape):
        mask = np.zeros(forecast_shape[1], dtype=bool)
        mask[region_of_interest] = True
        return mask

    def _process_mask_region(self, region_of_interest, forecast_shape):
        if (
            not isinstance(region_of_interest, np.ndarray)
            or region_of_interest.dtype != bool
        ):
            raise ValueError(
                "region_of_interest mask must be a numpy array of boolean type"
            )
        if region_of_interest.shape[0] != forecast_shape[1]:
            raise ValueError(
                "region_of_interest mask length must match the number of timesteps in the forecast"
            )
        return region_of_interest

    def _validate_and_maskify_region(self, region_of_interest, forecast_shape):
        if isinstance(region_of_interest, int):
            return self._process_int_region(region_of_interest, forecast_shape)
        elif isinstance(region_of_interest, list) and all(
            isinstance(i, int) for i in region_of_interest
        ):
            return self._process_list_region(region_of_interest, forecast_shape)
        elif isinstance(region_of_interest, slice):
            return self._process_slice_region(region_of_interest, forecast_shape)
        elif (
            isinstance(region_of_interest, np.ndarray)
            and region_of_interest.dtype == bool
        ):
            return self._process_mask_region(region_of_interest, forecast_shape)
        else:
            raise ValueError(
                "region_of_interest must be an int, a list of ints, a slice, or a boolean mask"
            )

    def _convert_float_weight_to_list(self, weight, region_of_interest_length):
        return [weight / region_of_interest_length] * region_of_interest_length

    def _validate_and_process_weight(
        self, weight, region_of_interest_length, complement_length
    ):
        if isinstance(weight, float):
            assert 0 <= weight <= 1, "Weight must be between 0 and 1"
            roi_weights = self._convert_float_weight_to_list(
                weight, region_of_interest_length
            )
        elif isinstance(weight, list):
            assert (
                len(weight) == region_of_interest_length
            ), "Weight list length must match region_of_interest length"
            assert sum(weight) <= 1, "Sum of weights must be at most 1"
            roi_weights = weight
        else:
            raise ValueError("Unsupported weight format")

        complement_weight = 1 - sum(roi_weights)
        complement_weights = [complement_weight / complement_length] * complement_length

        return roi_weights, complement_weights

    def _calculate_roi_weighted_metric(self, roi_metric_value, roi_weights):
        return np.sum(roi_weights) * roi_metric_value

    def _calculate_complement_weighted_metric(
        self, complement_metric_value, complement_weights
    ):
        return np.sum(complement_weights) * complement_metric_value

    def _calculate_combined_roi_metric(
        self, roi_metric_value, complement_metric_value, roi_weights, complement_weights
    ):
        roi_weighted_metric = self._calculate_roi_weighted_metric(
            roi_metric_value, roi_weights
        )
        complement_weighted_metric = self._calculate_complement_weighted_metric(
            complement_metric_value, complement_weights
        )
        return roi_weighted_metric + complement_weighted_metric


class RegionOfInterestMetric(BaseRegionOfInterestMetric):
    def __call__(self, target, forecast, region_of_interest, weight):
        roi_mask = self._validate_and_maskify_region(region_of_interest, forecast.shape)
        complement_mask = ~roi_mask

        roi_target = target[roi_mask]
        complement_target = target[complement_mask]

        roi_forecast = forecast[:, roi_mask]
        complement_forecast = forecast[:, complement_mask]

        roi_metric_value = self.metric(roi_target, roi_forecast)
        complement_metric_value = self.metric(complement_target, complement_forecast)

        roi_weights, complement_weights = self._validate_and_process_weight(
            weight, np.sum(roi_mask), np.sum(complement_mask)
        )

        combined_roi_metric = self._calculate_combined_roi_metric(
            roi_metric_value, complement_metric_value, roi_weights, complement_weights
        )
        target_range = np.max(target) - np.min(target)
        return combined_roi_metric / target_range


class ConstraintPenaltyMetric:
    def __init__(self, constraints=None, tolerance_percentage=0.05):
        self.constraints = constraints if constraints is not None else {}
        self.tolerance_percentage = tolerance_percentage
        self.scale_factor = np.log(2) / tolerance_percentage

    def _calculate_tolerance(self, target):
        target_range = np.max(target) - np.min(target)
        return self.tolerance_percentage * target_range

    def _calculate_penalty(self, target, forecast):
        penalty = 0
        tolerance = self._calculate_tolerance(target)

        if "min" in self.constraints:
            min_val = self.constraints["min"]
            penalty += np.sum(np.maximum(0, (min_val - tolerance) - forecast))

        if "max" in self.constraints:
            max_val = self.constraints["max"]
            penalty += np.sum(np.maximum(0, forecast - (max_val + tolerance)))

        return penalty * self.scale_factor


class RegionOfInterestConstraintMetric:
    def __init__(self, metric, constraints=None, tolerance_percentage=0.05):
        self.base_roi_metric = RegionOfInterestMetric(metric)
        self.constraint_penalty_metric = ConstraintPenaltyMetric(
            constraints, tolerance_percentage
        )

    def __call__(self, target, forecast, region_of_interest, weight):
        roi_mask = self.base_roi_metric._validate_and_maskify_region(
            region_of_interest, forecast.shape
        )

        roi_target = target[roi_mask]
        roi_forecast = forecast[:, roi_mask]

        roi_metric_value = self.base_roi_metric(
            target, forecast, region_of_interest, weight
        )
        penalty = self.constraint_penalty_metric._calculate_penalty(target, forecast)

        roi_metric_with_penalty = roi_metric_value * np.exp(penalty)

        return roi_metric_with_penalty


# Defining a sample metric function
def sample_metric(target, forecast):
    return crps_quantile(target=target, samples=forecast)[0].mean()


# Creating sample target and forecast ndarrays
target_data = np.array([1, 2, 3, 4, 5])
forecasts = [
    [1.0, 2.0, 3.0, 4.0, 5.0],  # Perfect forecast
    [1.1, 2.1, 3.1, 4.1, 5.1],  # Slightly off, respects constraints
    [5, 4, 3, 2, 1],  # Very off, respects constraints
    [1, 2, 3, 4, 6],  # Slightly off, does not respect constraints
    [6, 4, 2, 0, -2],  # Very off, does not respect constraints
    [1, 3, 4, 4, 5],  # Bad forecast in region of interest
    [2, 2, 3, 4, 4],  # Bad forecast in complement
]

tolerance_percentage = 0.05

# Initialize RegionOfInterestConstraintMetric with Min and Max Constraints
constraints = {"min": 1, "max": 5}
roi_constraint_metric = RegionOfInterestConstraintMetric(
    metric=sample_metric,
    constraints=constraints,
    tolerance_percentage=tolerance_percentage,
)


def plot_forecast(
    ax, target, forecast, region_of_interest, metric_value, constraints, title
):
    ax.plot(target, label="Target", marker="o", color="g")
    ax.plot(forecast.T, label="Forecast", linestyle="--", marker="x", alpha=0.7)

    min_val = constraints.get("min", None)
    max_val = constraints.get("max", None)
    tolerance = tolerance_percentage * (np.max(target) - np.min(target))

    if min_val is not None:
        ax.axhline(y=min_val, color="r", linestyle="-", label="Min Constraint")
        ax.axhline(
            y=min_val - tolerance,
            color="r",
            linestyle=":",
            label="Min Constraint - Tolerance",
        )

    if max_val is not None:
        ax.axhline(y=max_val, color="b", linestyle="-", label="Max Constraint")
        ax.axhline(
            y=max_val + tolerance,
            color="b",
            linestyle=":",
            label="Max Constraint + Tolerance",
        )

    ax.set_title(f"{title}\nMetric Value: {metric_value:.2f}")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.set_ylim([0, 10])


# Titles for each forecast example
titles = [
    "Perfect Forecast",
    "Slightly Off, Respects Constraints",
    "Very Off, Respects Constraints",
    "Slightly Off, Does Not Respect Constraints",
    "Very Off, Does Not Respect Constraints",
    "Bad Forecast in Region of Interest",
    "Bad Forecast in Complement",
]

fig, axs = plt.subplots(4, 2, figsize=(15, 20))
fig.tight_layout(pad=5.0)

region_of_interest = slice(1, 4)
weight = 0.91

# Loop through each forecast and plot
for i, forecast in enumerate(forecasts):
    ax = axs[i // 2, i % 2]
    forecast = np.array(forecast).reshape(1, -1)
    metric_value = roi_constraint_metric(
        target_data, forecast, region_of_interest, weight
    )
    plot_forecast(
        ax,
        target_data,
        forecast,
        region_of_interest,
        metric_value,
        constraints,
        titles[i],
    )
    print(f"Forecast {i + 1} Metric Value: {metric_value:.2f}")

# Hide the last empty subplot if needed
if len(forecasts) % 2 != 0:
    axs[-1, -1].axis("off")

plt.suptitle(
    r"$\text{ROI\_crps} = \left( \sum_{i=1}^n w_i \cdot \text{CRPS}_i \right) \times \exp\left(penalty_{constraint} * \frac{\log(2)}{\text{tolerance\_percentage}}\right)$",
    fontsize=14,
    y=0.995,
)
plt.tight_layout()

txt = """
         Each plot represents a single sample forecast.
         Target in solid green, forecast in dashed yellow
         The ground truth forecast is a line from 1 to 5. 
         The dashed line represents the forecast. 
         The red line represents the minimum constraint. 
         The blue line represents the maximum constraint. 
         The dotted lines represent the tolerance around the constraints 5(%). 
         The metric value is displayed in the title of each plot. 
         The metric value is calculated using the roi_metric * np.exp(penalty). 
         The roi_metric is calculated using CRPS (reduces to MAE for 1 sample). 
         The weight on the region of interest (timesteps 1,2,3) is 0.91.
         The penalty is calculated using the constraints.
         Hence, the metric values is roi_CRPS * exp(penalty).
        """

plt.figtext(0.52, 0.1, txt, wrap=True, horizontalalignment="left", fontsize=12)

plt.savefig(
    "/home/toolkit/starcaster/research-starcaster/benchmark/benchmark/metrics/forecast_comparison.png"
)
