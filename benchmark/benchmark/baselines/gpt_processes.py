"""
Open AI based LLM Process

"""

import logging
import numpy as np

from .base import Baseline
from ..config import (
    OPENAI_API_KEY,
    OPENAI_API_VERSION,
    OPENAI_AZURE_ENDPOINT,
    OPENAI_USE_AZURE,
)
from .utils import extract_html_tags


logger = logging.getLogger("GPT Processes")


class GPTForecaster(Baseline):
    """
    A simple baseline that uses any GPT model to produce forecastss

    Parameters:
    -----------
    model: str
        The name of the model to use for forecasting
    use_context: bool, default=True
        If True, use context in the prompt, otherwise ignore it
    fail_on_invalid: bool, default=True
        If True, raise an exception if an invalid sample is encountered
        in the forecast. Otherwise, print a warning and skip the sample.
    n_retries: int, default=3
        The number of retries to use in rejection sampling
    batch_size_on_retry: int, default=5
        The batch size to use on retries
    token_cost: dict, default=None
            The cost of tokens used in the API call. If provided, the cost of the API call will be estimated.
            Expected keys are "input" and "output" for the price of input and output tokens, respectively.

    """

    def __init__(
        self,
        model,
        use_context=True,
        fail_on_invalid=True,
        n_retries=3,
        batch_size_on_retry=5,
        token_cost: dict = None,
    ) -> None:
        self.model = model
        self.client = self.get_client()
        self.use_context = use_context
        self.fail_on_invalid = fail_on_invalid
        self.n_retries = n_retries
        self.batch_size_on_retry = batch_size_on_retry
        self.token_cost = token_cost
        self.total_cost = 0  # Accumulator for monetary value of queries

    def get_client(self):
        """
        Setup the OpenAI client based on configuration preferences

        """
        if OPENAI_USE_AZURE:
            logger.info("Using Azure OpenAI client.")
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=OPENAI_API_KEY,
                api_version=OPENAI_API_VERSION,
                azure_endpoint=OPENAI_AZURE_ENDPOINT,
            )
        else:
            logger.info("Using standard OpenAI client.")
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)

        return client

    def make_prompt(self, task_instance, max_digits=6):
        """
        Generate the prompt for the GPT model

        Notes:
        - Assumes a uni-variate time series

        """
        logger.info("Building prompt for GPT model.")

        # Extract time series data
        hist_time = task_instance.past_time.index.strftime("%Y-%m-%d %H:%M:%S").values
        hist_value = task_instance.past_time.values[:, -1]
        pred_time = task_instance.future_time.index.strftime("%Y-%m-%d %H:%M:%S").values
        history = "\n".join(
            f"({x}, {np.round(y, max_digits)})" for x, y in zip(hist_time, hist_value)
        )

        # Extract context
        context = ""
        if self.use_context:
            if task_instance.background:
                context += f"Background: {task_instance.background}\n"
            if task_instance.constraints:
                context += f"Constraints: {task_instance.constraints}\n"
            if task_instance.scenario:
                context += f"Scenario: {task_instance.scenario}\n"

        prompt = f"""
I have a time series forecasting task for you.

Here is some context about the task. Make sure to factor in any background knowledge,
satisfy any constraints, and respect any scenarios.
<context>
{context}
</context>

Here is a historical time series in (timestamp, value) format:
<history>
{history}
</history>

Now please predict the value at the following timestamps: {pred_time}.

Return the forecast in (timestamp, value) format in between <forecast> and </forecast> tags.
Do not include any other information (e.g., comments) in the forecast.

Example:
<history>
(t1, v1)
(t2, v2)
(t3, v3)
</history>
<forecast>
(t4, v4)
(t5, v5)
</forecast>

"""
        return prompt

    def __call__(self, task_instance, n_samples):
        """
        Infer forecasts from the GPT model

        Parameters:
        -----------
        task_instance: TimeSeriesTask
            The task instance to forecast
        n_samples: int
            The number of samples to generate
        n_retries: int
            The number of rejection sampling steps
        batch_size_on_retry: int
            The batch size to use on retries. This is useful to avoid asking for way too many samples
            from the openai API.

        Returns:
        --------
        samples: np.ndarray, shape [n_samples, time dimension, number of variables]
            The forecast samples. Note: only univariate is supported at the moment (number of variables = 1)

        """
        prompt = self.make_prompt(task_instance)
        messages = [
            {
                "role": "system",
                "content": "You are a useful forecasting assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        # Get forecast samples via rejection sampling until we have the desired number of samples
        # or until we run out of retries
        batch_size = n_samples
        total_tokens = {"input": 0, "output": 0}
        valid_forecasts = []
        n_retries = self.n_retries
        while len(valid_forecasts) < n_samples and n_retries > 0:
            logger.info(f"Requesting forecast of {batch_size} samples from GPT model.")
            chat_completion = self.client.chat.completions.create(
                model=self.model, n=batch_size, messages=messages
            )
            total_tokens["input"] += chat_completion.usage.prompt_tokens
            total_tokens["output"] += chat_completion.usage.completion_tokens

            logger.info("Parsing forecasts from completion.")
            for choice in chat_completion.choices:
                try:
                    # Extract forecast from completion
                    forecast = extract_html_tags(choice.message.content, ["forecast"])[
                        "forecast"
                    ][0]
                    forecast = forecast.replace("(", "").replace(")", "")
                    forecast = forecast.split("\n")
                    forecast = {
                        x.split(",")[0]
                        .replace("'", "")
                        .replace('"', ""): float(x.split(",")[1])
                        for x in forecast
                    }

                    # Get forecasted values at expected timestamps (will fail if model hallucinated timestamps, which is ok)
                    forecast = [
                        forecast[t]
                        for t in task_instance.future_time.index.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    ]

                    # Append forecast to list of valid forecasts
                    valid_forecasts.append(forecast)

                except Exception as e:
                    logger.info("Sample rejected due to invalid format.")
                    logger.debug(f"Rejection details: {e}")
                    logger.debug(f"Choice: {choice.message.content}")

            n_retries -= 1
            batch_size = self.batch_size_on_retry

            valid_forecasts = valid_forecasts[:n_samples]
            logger.info(f"Got {len(valid_forecasts)}/{n_samples} valid forecasts.")
            if len(valid_forecasts) < n_samples:
                logger.info(f"Remaining retries: {n_retries}.")

        # If we couldn't collect enough forecasts, raise exception if desired
        if self.fail_on_invalid and len(valid_forecasts) < n_samples:
            raise RuntimeError(
                f"Failed to get {n_samples} valid forecasts. Got {len(valid_forecasts)} instead."
            )

        # Estimate cost of API calls
        logger.info(f"Total tokens used: {total_tokens}")
        if self.token_cost is not None:
            input_cost = total_tokens["input"] / 1000 * self.token_cost["input"]
            output_cost = total_tokens["output"] / 1000 * self.token_cost["output"]
            current_cost = round(input_cost + output_cost, 2)
            logger.info(f"Forecast cost: {current_cost}$")
            self.total_cost += current_cost

        # Convert the list of valid forecasts to a numpy array
        samples = np.array(valid_forecasts)[:, :, None]

        return samples

    @property
    def cache_name(self):
        args_to_include = ["model", "use_context", "fail_on_invalid", "n_retries"]
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )
