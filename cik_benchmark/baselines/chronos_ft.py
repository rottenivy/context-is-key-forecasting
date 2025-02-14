import logging
logger = logging.getLogger(__file__)

import numpy as np
import time

from chronos import ChronosPipeline
from .chronos import ChronosForecaster
import torch

from transformers import set_seed


class ChronosFTForecaster(ChronosForecaster):
    __version__ = "0.1.0"  # Modification will trigger re-caching
    def forecast(
        self,
        task_instance,
        n_samples: int,
    ) -> np.ndarray:
        """
        This method allows a forecast to be done without requiring a complete BaseTask instance.
        This is primarily meant to be called inside a BaseTask constructor when doing rejection sampling or similar approaches.
        """
        # If there is no period, then disable the seasonal component of the model (seasonal_periods will be ignored)
        set_seed(self.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = ChronosPipeline.from_pretrained(
            f"rottenivy/chronos-t5-{self.model_size}-fine-tuned-{task_instance.__name__}",
            device_map=device,  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

        hist_values = torch.tensor(
            task_instance.past_time[[task_instance.past_time.columns[-1]]].values,
            dtype=torch.bfloat16,
        ).flatten()

        start_inference = time.time()
        # num_series, num_samples, num_timesteps
        model_preds = pipeline.predict(
            context=hist_values,
            prediction_length=len(task_instance.future_time),
            num_samples=n_samples,
            limit_prediction_length=False,
        )
        end_inference = time.time()

        # (1, num_samples, num_timesteps, num_series)
        model_preds = model_preds.permute(1, 2, 0)

        return model_preds.cpu().numpy(), {
            "inference_time": end_inference - start_inference
        }

