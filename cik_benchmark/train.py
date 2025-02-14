import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import traceback
from pprint import pprint

from collections import defaultdict
from functools import partial
from pathlib import Path

from . import ALL_TASKS
from .config import DEFAULT_N_SAMPLES
from .utils.cache import ResultCache, CacheMissError


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSeq2SeqLM  # or AutoModelForCausalLM if needed
import numpy as np
import os


# -------------------------------
# Define the train_task function.
# -------------------------------
def train_task(task_cls, seed, num_epochs=5, learning_rate=1e-4, batch_size=1, output_folder=None):
    """
    Fine-tunes the Chronos model (with extra textual prefix) on a given task.
    
    Arguments:
      task_cls: the task class (e.g., a subclass of ChronosForecaster_ExtraEmbed)
      seed: random seed for reproducibility.
      num_epochs: number of training epochs.
      learning_rate: optimizer learning rate.
      batch_size: training batch size.
      output_folder: (optional) folder to save the fine-tuned model.
    
    Returns:
      The fine-tuned model.
    """
    # Set seed (implement set_seed as needed)
    torch.manual_seed(seed)
    
    # Instantiate the task.
    task = task_cls(seed=seed)
    
    # Initialize your forecaster with extra textual input.
    # (Here ChronosForecaster_ExtraEmbed(model_size=...) is assumed to create a pipeline internally.)
    forecaster = task_cls(model_size=task.model_size)
    
    # Create a training dataset using your task instance and the pipeline's tokenizer.
    dataset = ChronosTrainDataset(task_instance=task, tokenizer=forecaster.tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Retrieve the model from your pipeline (CustomChronosModel).
    model = forecaster.pipeline.model  # type: CustomChronosModel
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    # For training, we also need target token IDs.
    # Use the tokenizer's label_input_transform to convert the numerical target to tokens.
    # Here we assume the label_input_transform returns (token_ids, attention_mask).
    target_token_ids, _ = forecaster.tokenizer.label_input_transform(
        dataset.target.unsqueeze(0), scale=torch.tensor([1.0])
    )
    target_token_ids = target_token_ids.to(model.device)  # shape: (batch, T_target_tokens)
    
    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # Prepare numerical context.
            # Assume context is a 1D tensor; _prepare_and_validate_context expects a tensor of shape (batch, T)
            context_tensor = forecaster._prepare_and_validate_context(batch["context"].unsqueeze(0))
            token_ids, attention_mask, scale = forecaster.tokenizer.context_input_transform(context_tensor)
            token_ids = token_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            
            # Obtain standard numerical embeddings.
            standard_embeds = model.model.shared(token_ids)  # shape: (batch, T+1, d_model)
            
            # Compute prefix embeddings using your prefix text encoder and the background text.
            prefix_text = batch["background"]  # a string
            prefix_embeds = forecaster.prefix_text_encoder(prefix_text)  # shape: (1, L_prefix, d_model)
            # Expand to batch size if necessary.
            if prefix_embeds.size(0) == 1 and standard_embeds.size(0) > 1:
                prefix_embeds = prefix_embeds.expand(standard_embeds.size(0), -1, -1)
            
            # Fuse: prepend prefix embeddings to standard embeddings.
            combined_embeds = torch.cat([prefix_embeds, standard_embeds], dim=1)
            # Update attention mask.
            batch_size_actual = attention_mask.size(0)
            prefix_mask = torch.ones((batch_size_actual, prefix_embeds.size(1)), dtype=attention_mask.dtype, device=model.device)
            combined_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
            # Forward pass.
            # Here we assume the model's forward, when given inputs_embeds, returns logits of shape (batch, seq_len, vocab_size)
            logits = model(
                input_ids=None,
                attention_mask=combined_attention_mask,
                prediction_length=target_token_ids.size(1),
                num_samples=1,  # use a single sample during training
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                inputs_embeds=combined_embeds,
            )
            # Reshape logits to (batch * seq_len, vocab_size) for loss computation.
            logits = logits.view(-1, logits.size(-1))
            target_flat = target_token_ids.view(-1)
            
            loss = loss_fn(logits, target_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    # Optionally, save the fine-tuned model.
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        save_path = output_folder / f"{task_cls.__name__}_fine_tuned.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return model
