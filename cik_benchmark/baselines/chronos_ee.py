import logging
logger = logging.getLogger(__file__)
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import time

from.chronos import ChronosForecaster
from chronos import ChronosConfig, ChronosModel, ChronosPipeline
import torch
import torch.nn as nn

from transformers import set_seed
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    T5Tokenizer,
    T5ForConditionalGeneration
)

class PrefixTextEncoder(nn.Module):
    def __init__(self, model_name: str = "google/t5-efficient-tiny", target_d_model: int = 256):
        """
        A simple module that converts a text string into embeddings.
        It uses T5’s tokenizer to tokenize the text and T5’s shared embedding
        layer to map token IDs into embeddings. If the embedding dimension of T5
        does not match target_d_model, a linear projection is applied.
        """
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.embeddings = T5ForConditionalGeneration.from_pretrained(model_name).shared
        self.t5_hidden_size = self.embeddings.embedding_dim
        
        # If needed, project the embeddings to target_d_model.
        if self.t5_hidden_size != target_d_model:
            self.proj = nn.Linear(self.t5_hidden_size, target_d_model)
        else:
            self.proj = None

    def forward(self, text_input: str):
        # Tokenize the text input.
        encoded = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(next(self.parameters()).device)
        # Obtain embeddings from T5's shared embedding layer.
        embeds = self.embeddings(input_ids)  # shape: (batch_size, seq_len, t5_hidden_size)
        if self.proj is not None:
            embeds = self.proj(embeds)  # shape: (batch_size, seq_len, target_d_model)
        return embeds


class CustomChronosModel(ChronosModel):
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        inputs_embeds: torch.Tensor = None  # new optional argument
    ) -> torch.Tensor:
        """
        Difference: the generate() method will use ``inputs_embeds`` if provided.

        Predict future sample tokens for the given token sequences.

        Arguments ``prediction_length``, ``num_samples``, ``temperature``,
        ``top_k``, ``top_p`` can be used to customize the model inference,
        and default to the corresponding attributes in ``self.config`` if
        not provided.

        Returns
        -------
        samples
            A tensor of integers, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        # Check that exactly one of input_ids or inputs_embeds is provided.
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds.")

        if prediction_length is None:
            prediction_length = self.config.prediction_length
        if num_samples is None:
            num_samples = self.config.num_samples
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if top_p is None:
            top_p = self.config.top_p

        if inputs_embeds is not None:
            preds = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(
                    min_new_tokens=prediction_length,
                    max_new_tokens=prediction_length,
                    do_sample=True,
                    num_return_sequences=num_samples,
                    eos_token_id=self.config.eos_token_id,
                    pad_token_id=self.config.pad_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ),
            )
        else:
            preds = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(
                    min_new_tokens=prediction_length,
                    max_new_tokens=prediction_length,
                    do_sample=True,
                    num_return_sequences=num_samples,
                    eos_token_id=self.config.eos_token_id,
                    pad_token_id=self.config.pad_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ),
            )

        if self.config.model_type == "seq2seq":
            preds = preds[..., 1:]  # remove the decoder start token
        else:
            assert self.config.model_type == "causal"
            assert preds.size(-1) == input_ids.size(-1) + prediction_length
            preds = preds[..., -prediction_length:]

        return preds.reshape(input_ids.size(0) if input_ids is not None else inputs_embeds.size(0),
                             num_samples,
                             -1)


class CustomChronosPipeline(ChronosPipeline):
    def __init__(self, tokenizer, model):
        super().__init__(tokenizer, model)
        self.tokenizer = tokenizer
        self.model = model
        # Initialize the prefix text encoder.
        # Here we assume the target embedding dimension matches Chronos' model dimension.
        self.prefix_text_encoder = PrefixTextEncoder(
            model_name="google/t5-efficient-tiny", target_d_model=self.model.model.model_dim
        ).to(self.model.device)

    def predict(  # type: ignore[override]
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        limit_prediction_length: bool = False,
        prefix_text: Optional[str] = None  # new parameter for textual prefix
    ) -> torch.Tensor:
        """
        Difference: Support for extra embeddings in the pipeline.

        Get forecasts for the given time series.

        Refer to the base method (``ChronosPipeline.predict``)
        for details on shared parameters.

        Additional parameters
        ---------------------
        num_samples
            Number of sample paths to predict. Defaults to what
            specified in ``self.model.config``.
        temperature
            Temperature to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        top_k
            Top-k parameter to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        top_p
            Top-p parameter to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. False by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        samples
            Tensor of sample forecasts, of shape
            (batch_size, num_samples, prediction_length).
        """
        context_tensor = self._prepare_and_validate_context(context=context)

        if prediction_length is None:
            prediction_length = self.model.config.prediction_length

        if prediction_length > self.model.config.prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {self.model.config.prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            logger.warning(msg)

        # If a textual prefix is provided and this is the first iteration, compute prefix embeddings.
        if prefix_text is not None:
            prefix_embeds = self.prefix_text_encoder(prefix_text)  # shape: (1, L_prefix, d_model) or (batch, L_prefix, d_model)
            # Update the attention mask: create a mask for the prefix (all ones) and concatenate.
            prefix_length = prefix_embeds.size(1)

        predictions = []
        remaining = prediction_length

        while remaining > 0:
            token_ids, attention_mask, scale = self.tokenizer.context_input_transform(
                context_tensor
            )

            # Compute the standard embeddings from the model's shared embedding layer.
            # (Assumes self.model.model.shared is the shared embedding layer.)
            series_embeds = self.model.model.shared(token_ids.to(self.model.device))
            combined_embeds = torch.cat([prefix_embeds, series_embeds], dim=1)
            batch_size = attention_mask.size(0)
            prefix_length = prefix_embeds.size(1)
            prefix_mask = torch.ones((batch_size, prefix_length), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            samples = self.model(
                None,
                attention_mask.to(self.model.device),
                min(remaining, self.model.config.prediction_length),
                num_samples,
                temperature,
                top_k,
                top_p,
                combined_embeds,
            )
            prediction = self.tokenizer.output_transform(
                samples.to(scale.device), scale
            )

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            context_tensor = torch.cat(
                [context_tensor, prediction.median(dim=1).values], dim=-1
            )

        return torch.cat(predictions, dim=-1).to(dtype=torch.float32, device="cpu")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """

        config = AutoConfig.from_pretrained(*args, **kwargs)

        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        chronos_config = ChronosConfig(**config.chronos_config)

        if chronos_config.model_type == "seq2seq":
            inner_model = AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
        else:
            assert chronos_config.model_type == "causal"
            inner_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        return cls(
            tokenizer=chronos_config.create_tokenizer(),
            model=CustomChronosModel(config=chronos_config, model=inner_model),
        )


class ChronosForecaster_ExtraEmbed(ChronosForecaster):
    __version__ = "0.1.1"  # update version to trigger re-caching if needed

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
        pipeline = CustomChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{self.model_size}",
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
            prefix_text=task_instance.get_background()
        )
        end_inference = time.time()

        # (1, num_samples, num_timesteps, num_series)
        model_preds = model_preds.permute(1, 2, 0)

        return model_preds.cpu().numpy(), {
            "inference_time": end_inference - start_inference
        }
