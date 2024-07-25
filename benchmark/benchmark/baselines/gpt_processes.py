"""
Open AI based LLM Process

"""

import logging
import numpy as np

from ..config import (
    OPENAI_API_KEY,
    OPENAI_API_VERSION,
    OPENAI_AZURE_ENDPOINT,
    OPENAI_USE_AZURE,
)
from .utils import extract_html_tags


logger = logging.getLogger("GPT Processes")


class GPTForecaster:
    """
    A simple baseline that uses any GPT model to produce forecastss

    Parameters:
    -----------
    model: str
        The name of the model to use for forecasting
    use_context: bool
        If True, use context in the prompt, otherwise ignore it
    fail_on_invalid: bool
        If True, raise an exception if an invalid sample is encountered
        in the forecast. Otherwise, print a warning and skip the sample.

    """

    def __init__(self, model, use_context=True, fail_on_invalid=True) -> None:
        self.model = model
        self.client = self.get_client()
        self.use_context = use_context
        self.fail_on_invalid = fail_on_invalid

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

    def make_prompt(self, task_instance):
        """
        Generate the prompt for the GPT model

        Notes:
        - Assumes a uni-variate time series

        """
        logger.info("Building prompt for GPT model.")

        # Extract time series data
        hist_time = task_instance.past_time.index.strftime("%Y-%m-%d %H:%M:%S")
        hist_value = task_instance.past_time.values[:, 0]
        pred_time = task_instance.future_time.index.strftime("%Y-%m-%d %H:%M:%S")
        history = "\n".join(
            f"({x}, {np.round(y, 2)})" for x, y in zip(hist_time, hist_value)
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
        assert (
            task_instance.past_time.shape[-1] == 1
        ), "Only uni-variate time series are supported at the moment."

        prompt = self.make_prompt(task_instance)

        logger.info(f"Requesting forecast of {n_samples} samples from GPT model.")
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            n=n_samples,
            messages=[
                {
                    "role": "system",
                    "content": "You are a useful forecasting assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        logger.info("Parsing forecasts from completion.")
        valid_forecasts = []
        for choice in chat_completion.choices:
            try:
                # Extract forecast from completion
                forecast = extract_html_tags(choice.message.content, ["forecast"])[
                    "forecast"
                ][0]
                forecast = forecast.replace("(", "").replace(")", "")
                forecast = forecast.split("\n")
                forecast = {x.split(",")[0]: float(x.split(",")[1]) for x in forecast}

                # Get forecasted values at expected timestamps (will fail if model hallucinated timestamps, which is ok)
                forecast = [
                    forecast[t]
                    for t in task_instance.future_time.index.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ]
                valid_forecasts.append(forecast)
            except Exception as e:
                if self.fail_on_invalid:
                    raise e
                logger.info(
                    "Error: No forecast or incorrect format. Skipping this sample."
                )
                logger.debug(f"Exception details: {e}")
                logger.debug(f"Forecast: {forecast}")
                logger.debug(f"Choice: {choice.message.content}")

        # Convert the list of valid forecasts to a numpy array
        logger.info(f"Got {len(valid_forecasts)} valid forecasts.")
        samples = np.array(valid_forecasts)[:, :, None]

        return samples
