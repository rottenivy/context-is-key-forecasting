import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_task(task):
    """
    Plots a task's numerical data and returns the figure.

    Parameters:
    -----------
    task: BaseTask
        The task to plot

    Returns:
    --------
    fig: matplotlib.figure.Figure
        The figure containing the plot

    """
    plt.figure()

    # Prepare data
    past_timesteps = task.past_time.index
    past_values = task.past_time.to_numpy()[:, 0]
    future_timesteps = task.future_time.index
    future_values = task.future_time.to_numpy()[:, 0]

    # The fill_between method is only ok with pd.DatetimeIndex
    if isinstance(past_timesteps, pd.PeriodIndex):
        past_timesteps = past_timesteps.to_timestamp()
    if isinstance(future_timesteps, pd.PeriodIndex):
        future_timesteps = future_timesteps.to_timestamp()

    timesteps = np.concatenate([past_timesteps, future_timesteps])
    values = np.concatenate([past_values, future_values])

    # Plot the history
    plt.plot(
        past_timesteps,
        past_values,
        color=(0, 0, 0),
        linewidth=2,
        zorder=5,
        label="History",
    )

    # Plot the forecast
    plt.plot(
        future_timesteps,
        future_values,
        color="orange",
        linewidth=2,
        zorder=5,
        label="Ground Truth",
    )

    # Shade the entire future region in light green to indicate the forecast region
    plt.fill_between(
        future_timesteps,
        -99999,  # Use the current minimum y-limit
        99999,  # Use the current maximum y-limit
        facecolor=(0.9, 1, 0.9),
        interpolate=True,
        label="Forecast",
        zorder=0,
    )
    # Set the plot limits to ensure no white space around the shading
    plt.ylim(values.min(), values.max())
    plt.xlim(timesteps[0], timesteps[-1])

    # Minor style tweaks
    plt.xticks(rotation=90)
    plt.tight_layout()

    return plt.gcf()
