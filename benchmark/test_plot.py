from pathlib import Path

from benchmark.predictable_grocer_shocks import PredictableGrocerPersistentShockUnivariateTask
from benchmark.tasks.causal_chambers import ExplicitPressureFromSpeedTask
from benchmark.baselines.statsmodels import (
    ExponentialSmoothingForecaster,
)

from benchmark.evaluation import evaluate_task

model = ExponentialSmoothingForecaster()

results = evaluate_task(
    ExplicitPressureFromSpeedTask,
    10,
    model,
    10,
    output_folder=Path("./plots/ExplicitPressureFromSpeedTask/"),
)

results = evaluate_task(
    PredictableGrocerPersistentShockUnivariateTask,
    0,
    model,
    10,
    output_folder=Path("./plots/PredictableGrocerPersistentShockUnivariateTask/"),
)