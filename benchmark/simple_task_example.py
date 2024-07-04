import matplotlib.pyplot as plt
import numpy as np
import textwrap

from benchmark.misleading_history import PeriodicSensorMaintenanceTask


for i in range(5):
    # Create a random instance of the task
    task = PeriodicSensorMaintenanceTask()

    # Evaluate a random forecast
    print("Score:", task.evaluate(np.random.rand(50, task.future_time.shape[0], 1)))

    # Plot the history and future series with the background information
    plt.clf()
    task.past_time.plot(label="History")
    task.future_time.plot(label="Future")
    plt.title("\n".join(textwrap.wrap(task.background, width=40)))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"periodic_sensor_maintenance_task_{i}.png", bbox_inches="tight")
