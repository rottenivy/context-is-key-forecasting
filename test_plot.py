from pathlib import Path

from cik_benchmark.predictable_grocer_shocks import (
    PredictableGrocerPersistentShockUnivariateTask,
)
from cik_benchmark.tasks.causal_chambers import (
    ExplicitPressureFromSpeedTask,
    SpeedFromLoadTask,
)
from cik_benchmark.baselines.statsmodels import (
    ExponentialSmoothingForecaster,
)

from cik_benchmark.evaluation import evaluate_task

model = ExponentialSmoothingForecaster()

results = evaluate_task(
    ExplicitPressureFromSpeedTask,
    10,
    model,
    10,
    output_folder=Path("./plots/ExplicitPressureFromSpeedTask/"),
)

print(results)

results = evaluate_task(
    SpeedFromLoadTask,
    5,
    model,
    10,
    output_folder=Path("./plots/SpeedFromLoadTask/"),
)

print(results)

results = evaluate_task(
    PredictableGrocerPersistentShockUnivariateTask,
    0,
    model,
    10,
    output_folder=Path("./plots/PredictableGrocerPersistentShockUnivariateTask/"),
)

print(results)
