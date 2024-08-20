import gc
import logging
import numpy as np
import os
import pickle
import tempfile
import torch

from llm_processes.hf_api import get_model_and_tokenizer
from llm_processes.parse_args import llm_map, parse_command_line
from llm_processes.run_llm_process import run_llm_process

from .base import Baseline


class LLMPForecaster(Baseline):
    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(self, llm_type, use_context=True):
        f"""
        Get predictions from LLM processes

        Parameters:
        -----------
        llm_type: str
            Type of LLM model to use. Options are: {llm_map.keys()}
        use_context: bool
            Whether to include context in the prompt

        Notes:
        ------
        * TODO: No multivariate support
        * By default, the model is set in autoregressive mode
        
        """
        self.llm_type = llm_type
        self.use_context = use_context

        # LLMP relies on the disk to store input/outputs. We parameterize a few paths
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self.tmpdir.name
        self.input_data_path = f"{self.output_dir}/input_data.tmp"
        self.experiment_name = "llmp_runner"
        self.output_data_path = f"{self.output_dir}/{self.experiment_name}.pkl"

        self.llmp_args = {
            "--llm_type": llm_type,
            "--data_path": self.input_data_path,
            "--forecast": "true",
            "--autoregressive": "true",
            "--output_dir": self.output_dir,
            "--experiment_name": self.experiment_name,
            "--num_samples": None,  # This is set in the __call__ method
        }
        # TODO: Are there other args we should set?

        # Load the model and tokenizer
        logging.info("Loading model and tokenizer...")
        self.model, self.tokenizer = get_model_and_tokenizer(
            llm_path=None, llm_type=self.llmp_args["--llm_type"]
        )

    def _prepare_data(self, task_instance):
        """
        Formats the data and pickles it to be consumed by the LLMP process

        Parameters:
        task_instance: BaseTask
            Task instance for which to forecast

        """
        logging.info("Preparing data for LLMP...")
        llmp_data = {}
        # Take the last column of the dataframe (the forecast variable), since we are only modelling the forecast variable for now
        past_time = task_instance.past_time[task_instance.past_time.columns[-1]]
        future_time = task_instance.future_time[task_instance.future_time.columns[-1]]

        # XXX: Convert to unix timestamp since LLMP expects numbers for x and y
        llmp_data["x_train"] = past_time.index.astype(int).values // 10**9
        llmp_data["x_test"] = future_time.index.astype(int).values // 10**9
        llmp_data["x_true"] = np.hstack((llmp_data["x_train"], llmp_data["x_test"]))
        llmp_data["y_train"] = past_time.values
        llmp_data["y_test"] = future_time.values
        llmp_data["y_true"] = np.hstack((llmp_data["y_train"], llmp_data["y_test"]))
        with open(self.llmp_args["--data_path"], "wb") as f:
            pickle.dump(llmp_data, f)

    def _load_results(self):
        """
        Load results from pickle file outputted by LLM process

        """
        # TODO: There are plots that are auto-generated and we currently ignore them
        logging.info("Loading results from LLMP...")
        with open(self.output_data_path, "rb") as f:
            results = pickle.load(f)

            samples = np.empty((len(results["gen"]), len(results["gen"][0])))

            # Loop over every time point in the forecast
            for i, point_samples in enumerate(results["gen"]):
                # Loop over every sample for each time point
                for j, sample in enumerate(point_samples):
                    samples[i][j] = float(sample.split("\n")[0])

        return samples

    def _make_prompt(self, task_instance):
        """
        Formats the prompt and adds it to the LLMP arguments

        """
        prompt = f"""
Forecast the future values of this time series, while considering the following 
background knowledge, scenario, and constraints.

Background knowledge:
{task_instance.background}

Scenario:
{task_instance.scenario}

Constraints:
{task_instance.constraints}

"""
        self.llmp_args["--prefix"] = prompt

    def __call__(self, task_instance, n_samples):
        """
        Perform inference with LLMP

        Parameters:
        -----------
        task_instance: BaseTask
            Task instance for which to perform inference
        n_samples: int
            Number of samples to draw from the model

        Returns:
        --------
        samples: np.ndarray, shape=(n_samples, task_instance.future_time.shape[0], 1)
            Samples drawn from the model

        """
        logging.info("Forecasting with LLMP...")
        self._prepare_data(task_instance)

        logging.info("Preparing prompt...")
        if self.use_context:
            self._make_prompt(task_instance)
        else:
            if "--prefix" in self.llmp_args:
                del self.llmp_args["--prefix"]

        # Set number of samples
        self.llmp_args["--num_samples"] = str(n_samples)

        # Run LLMP
        logging.info("Running LLM process...")
        llmp_args = parse_command_line(
            [item for pair in self.llmp_args.items() for item in pair]
        )
        run_llm_process(args=llmp_args, model=self.model, tokenizer=self.tokenizer)

        # Get results
        samples = self._load_results()
        return samples.transpose()[
            :, :, None
        ]  # XXX: Would need to be adapted when we expand to multivariate

    def __del__(self):
        """
        Clean up the temporary directory

        """
        self.tmpdir.cleanup()

        # Clean up CPU/GPU memory
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @property
    def cache_name(self):
        args_to_include = ["llm_type", "use_context", "fail_on_invalid", "n_retries"]
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )
