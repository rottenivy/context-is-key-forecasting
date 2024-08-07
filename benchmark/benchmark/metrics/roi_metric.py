import numpy as np
from benchmark.metrics.crps import crps_quantile

import matplotlib.pyplot as plt


def crps_by_quantile_mean(target, samples):
    if len(target) > 0:
        return crps_quantile(target, samples)[0].mean() / len(target)
    else:
        return 0


def region_of_interest_constraint_metric(
    target,
    forecast,
    region_of_interest,
    roi_weight,
    roi_metric=crps_by_quantile_mean,
    constraints={},
    tolerance_percentage=0.05,
):
    """
    Function to calculate a combined metric that considers a region of interest and constraints.

    Parameters:
    ----------
    target: np.ndarray
        The target values. (n_timesteps,)
    forecast: np.ndarray
        The forecast values. (n_samples, n_timesteps)
    region_of_interest: int, list of ints, slice, or boolean mask
        The region of interest to apply the roi_metric to.
    roi_weight: float
        The weight to apply to the region of interest.
    roi_metric: function, optional
        The metric to apply to the region of interest. Default is crps_by_quantile_mean.
    constraints: dict, optional
        A dictionary containing the constraints. The keys are "min" and "max". Default is None.
    tolerance_percentage: float, optional
        The percentage of the target range to use as a tolerance for the constraints. Default is 0.05.

    Returns:
    --------
    combined_roi_metric_with_penalty: float
        The combined metric value with the constraint penalty applied.
    """

    combined_roi_metric = calculate_combined_roi_metric(
        target, forecast, region_of_interest, roi_weight, roi_metric
    )
    constraint_penalty = calculate_constraint_penalty(
        target, forecast, constraints, tolerance_percentage
    )

    combined_roi_metric_with_penalty = combined_roi_metric * np.exp(constraint_penalty)

    return combined_roi_metric_with_penalty


def calculate_combined_roi_metric(
    target,
    forecast,
    region_of_interest,
    roi_weight,
    roi_metric,
    divide_by_target_range=True,
):
    """
    Calculate the combined metric for the region of interest and its complement.

    Parameters:
    -----------
    target: np.ndarray
        The target values. (n_timesteps,)
    forecast: np.ndarray
        The forecast values. (n_samples, n_timesteps)
    region_of_interest: int, list of ints, slice, or boolean mask
        The region of interest to apply the roi_metric to.
    roi_weight: float
        The weight to apply to the region of interest.
    roi_metric: function
        The metric to apply to the region of interest.
    divide_by_target_range: bool, optional
        Whether to divide the combined metric by the target range. Default is True.

    Returns:
    --------
    combined_roi_metric: float
        The combined metric value for the region of interest and its complement.
    """
    roi_mask = format_roi_mask(region_of_interest, forecast.shape)

    weighted_roi_metric = calculate_masked_metric(
        target, forecast, roi_mask, roi_weight, roi_metric
    )

    weighted_complement_metric = calculate_masked_metric(
        target, forecast, ~roi_mask, (1 - roi_weight), roi_metric
    )
    combined_roi_metric = weighted_roi_metric + weighted_complement_metric

    if divide_by_target_range:
        target_range = np.max(target) - np.min(target)
        return combined_roi_metric / target_range
    else:
        return combined_roi_metric


def calculate_masked_metric(target, forecast, roi_mask, roi_weight, roi_metric):
    """
    Calculate the weighted metric for a masked region of interest.
    Parameters:
    -----------
    target: np.ndarray
        The target values. (n_timesteps,)
    forecast: np.ndarray
        The forecast values. (n_samples, n_timesteps)
    roi_mask: np.ndarray
        The boolean mask for the region of interest.
    roi_weight: float
        The weight to apply to the region of interest.
    roi_metric: function
        The metric to apply to the region of interest.
    Returns:
    --------
    weighted_roi_metric: float
        The weighted metric value for the region of interest.

    """
    roi_target = target[roi_mask]
    roi_forecast = forecast[:, roi_mask]
    roi_metric = roi_metric(target=roi_target, samples=roi_forecast)
    weighted_roi_metric = roi_weight * roi_metric
    return weighted_roi_metric


def calculate_constraint_penalty(
    target, forecast, constraints, tolerance_percentage, scale_factor=np.log(2)
):
    """
    Calculate the penalty for violating constraints.

    Parameters:
    -----------
    target: np.ndarray
        The target values. (n_timesteps,)
    forecast: np.ndarray
        The forecast values. (n_samples, n_timesteps)
    constraints: dict
        A dictionary containing the constraints. The keys are "min" and "max".
    tolerance_percentage: float
        The percentage of the target range to use as a tolerance for the constraints.
    scale_factor: float, optional
        The scale factor to apply to the penalty. Default is np.log(2).

    Returns:
    --------
    penalty: float
        The penalty value for violating constraints.
    """
    penalties = []
    constraint_tolerance = (np.max(target) - np.min(target)) * tolerance_percentage

    for sample_forecast in forecast:
        penalty = 0
        if "min" in constraints:
            min_val = constraints["min"]
            penalty += np.sum(
                np.maximum(0, (min_val - constraint_tolerance) - sample_forecast)
            )

        if "max" in constraints:
            max_val = constraints["max"]
            penalty += np.sum(
                np.maximum(0, sample_forecast - (max_val + constraint_tolerance))
            )

        penalty = penalty / len(target)

        penalties.append(penalty)

    average_penalty = np.mean(penalties)
    if tolerance_percentage == 0:
        return average_penalty * scale_factor
    else:
        return average_penalty * scale_factor


def format_roi_mask(region_of_interest, forecast_shape):
    """
    Formats the region of interest mask based on the given inputs.

    Parameters:
    ----------
    region_of_interest (int, list, slice, np.ndarray): The region of interest mask.
        - If an int, a mask with a single True value at the specified index is returned.
        - If a list of ints, a mask with True values at the specified indices is returned.
        - If a slice, a mask with True values within the specified range is returned.
        - If a boolean np.ndarray, the mask is returned as is.
    forecast_shape (tuple): The shape of the forecast.

    Returns:
    -------
        np.ndarray: The formatted region of interest mask.

    """
    if isinstance(region_of_interest, int):
        mask = np.zeros(forecast_shape[1], dtype=bool)
        mask[region_of_interest] = True
        return mask
    elif isinstance(region_of_interest, list) and all(
        isinstance(i, int) for i in region_of_interest
    ):
        mask = np.zeros(forecast_shape[1], dtype=bool)
        for i in region_of_interest:
            mask[i] = True
        return mask
    elif isinstance(region_of_interest, slice):
        mask = np.zeros(forecast_shape[1], dtype=bool)
        mask[region_of_interest] = True
        return mask
    elif (
        isinstance(region_of_interest, np.ndarray) and region_of_interest.dtype == bool
    ):
        if region_of_interest.shape[0] != forecast_shape[1]:
            raise ValueError(
                "region_of_interest mask length must match the number of timesteps in the forecast"
            )
        return region_of_interest
    else:
        raise ValueError(
            "region_of_interest must be an int, a list of ints, a slice, or a boolean mask"
        )


if __name__ == "__main__":

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

    # Creating sample target and forecast ndarrays
    target_data = np.array([1, 2, 3, 4, 5])
    # Original forecasts
    forecasts = [
        [1.0, 2.0, 3.0, 4.0, 5.0],  # Perfect forecast
        [1.1, 2.1, 3.1, 4.1, 5.1],  # Slightly off, respects constraints
        [5, 4, 3, 2, 1],  # Very off, respects constraints
        [1, 2, 3, 4, 6],  # Slightly off, does not respect constraints
        [6, 4, 2, 0, -2],  # Very off, does not respect constraints
        [1, 3, 4, 4, 5],  # Bad forecast in region of interest
        [2, 2, 3, 4, 4],  # Bad forecast in complement
    ]

    # Number of samples per forecast
    n_samples = 3

    # Generate multiple samples per forecast
    multi_sample_forecasts = np.array(
        [
            np.array(forecast)
            + np.random.normal(0, 0.01, size=(n_samples, len(forecast)))
            for forecast in forecasts
        ]
    )

    # Transpose to (n_forecasts, n_samples, n_timesteps)
    forecasts = np.transpose(multi_sample_forecasts, (0, 1, 2))

    # Initialize RegionOfInterestConstraintMetric with Min and Max Constraints
    constraints = {"min": 1, "max": 5}
    region_of_interest = slice(1, 4)
    roi_weight = 0.91
    tolerance_percentage = 0.1

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

    # Loop through each forecast and plot
    for i, forecast in enumerate(forecasts):
        metric_value = region_of_interest_constraint_metric(
            target=target_data,
            forecast=forecast,
            region_of_interest=region_of_interest,
            roi_weight=roi_weight,
            roi_metric=crps_by_quantile_mean,
            constraints=constraints,
            tolerance_percentage=tolerance_percentage,
        )
        ax = axs[i // 2, i % 2]
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
        "/home/toolkit/starcaster/research-starcaster/benchmark/benchmark/metrics/forecast_comparison_functions.png"
    )
