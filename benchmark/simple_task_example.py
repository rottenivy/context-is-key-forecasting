import matplotlib.pyplot as plt
import pandas as pd
import textwrap

from benchmark.sensor_maintenance import SensorMaintenanceInPredictionTask


def plot_forecast_univariate(task, filename):
    """
    Plot the first variable of a forecast.

    Parameters:
    -----------
    task: a BaseTask
        The task associated with the forecast
    filename: Pathlike
        Where to save the figure
    """
    past_timesteps = task.past_time.index
    past_values = task.past_time.to_numpy()[:, -1]
    future_timesteps = task.future_time.index
    future_values = task.future_time.to_numpy()[:, -1]

    # The fill_between method is only ok with pd.DatetimeIndex
    if isinstance(past_timesteps, pd.PeriodIndex):
        past_timesteps = past_timesteps.to_timestamp()
    if isinstance(future_timesteps, pd.PeriodIndex):
        future_timesteps = future_timesteps.to_timestamp()

    plt.figure(figsize=(15, 15))

    plt.plot(
        past_timesteps,
        past_values,
        color=(1, 0, 0),
        linewidth=2,
        zorder=5,
        label="history",
    )
    plt.plot(
        future_timesteps,
        future_values,
        color=(0, 0, 1),
        linewidth=2,
        zorder=5,
        label="prediction",
    )
    plt.xticks(fontsize=7)

    plt.legend()

    plt.title(
        "\n".join(textwrap.wrap(task.scenario, width=40))
    )  # Change this to task.background or task.constraint depending on task
    plt.savefig(filename)


task = SensorMaintenanceInPredictionTask()  # Change this to required task
plot_forecast_univariate(task, f"task_example_plotting.png")
