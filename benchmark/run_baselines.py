"""
Run all baselines on all tasks and save the results to a Pandas dataframe.

"""

import argparse
import json
import inspect
import logging
import numpy as np
import pandas as pd

from pathlib import Path

from benchmark.baselines.gpt_processes import GPTForecaster
from benchmark.baselines.lag_llama import lag_llama
from benchmark.baselines.llm_processes import LLMPForecaster
from benchmark.baselines.naive import oracle_baseline, random_baseline
from benchmark.baselines.statsmodels import (
    ExponentialSmoothingForecaster,
)
from benchmark.evaluation import evaluate_all_tasks


logging.basicConfig(level=logging.INFO)


def experiment_naive(
    n_samples, output_folder, max_parallel=None, skip_cache_miss=False
):
    """
    Naive baselines (random and oracle)

    """
    results = []
    results.append(
        (
            "random",
            evaluate_all_tasks(
                random_baseline,
                n_samples=n_samples,
                output_folder=f"{output_folder}/random/",
                max_parallel=max_parallel,
                skip_cache_miss=skip_cache_miss,
            ),
        )
    )
    results.append(
        (
            "oracle",
            evaluate_all_tasks(
                oracle_baseline,
                n_samples=n_samples,
                output_folder=f"{output_folder}/oracle/",
                max_parallel=max_parallel,
                skip_cache_miss=skip_cache_miss,
            ),
        )
    )
    return results, {}


def experiment_lag_llama(
    n_samples, output_folder, max_parallel=10, skip_cache_miss=False
):
    """
    Lag LLAMA baseline

    """
    results = evaluate_all_tasks(
        lag_llama,
        n_samples=n_samples,
        output_folder=f"{output_folder}/lag_llama/",
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
    )
    return results, {}


def experiment_statsmodels(
    n_samples, output_folder, max_parallel=None, skip_cache_miss=False
):
    """
    Statsmodels baselines (Exponential Smoothing)

    """
    return (
        evaluate_all_tasks(
            ExponentialSmoothingForecaster(),
            n_samples=n_samples,
            output_folder=f"{output_folder}/exp_smoothing/",
            max_parallel=max_parallel,
            skip_cache_miss=skip_cache_miss,
        ),
        {},
    )


def experiment_gpt(
    llm, use_context, n_samples, output_folder, max_parallel=1, skip_cache_miss=False
):
    """
    GPT baselines

    """
    # Costs per 1000 tokens
    openai_costs = {
        "gpt-4o": {"input": 0.005, "output": 0.015},  # Same price Azure and OpenAI
        "gpt-35-turbo": {"input": 0.002, "output": 0.002},
        "gpt-3.5-turbo": {"input": 0.003, "output": 0.006},  # OpenAI API
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # OpenAI API
    }
    if llm not in openai_costs:
        raise ValueError(f"Invalid model: {llm} -- Not in cost dictionary")

    gpt_forecaster = GPTForecaster(
        model=llm, use_context=use_context, token_cost=openai_costs[llm]
    )
    results = evaluate_all_tasks(
        gpt_forecaster,
        n_samples=n_samples,
        output_folder=f"{output_folder}/{gpt_forecaster.cache_name}",
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
    )
    total_cost = gpt_forecaster.total_cost
    del gpt_forecaster

    return results, {"total_cost": total_cost}


def experiment_llmp(
    llm, use_context, n_samples, output_folder, max_parallel=1, skip_cache_miss=False
):
    """
    LLM Process baselines

    """
    llmp_forecaster = LLMPForecaster(llm_type=llm, use_context=use_context)
    return (
        evaluate_all_tasks(
            llmp_forecaster,
            n_samples=n_samples,
            output_folder=f"{output_folder}/{llmp_forecaster.cache_name}",
            max_parallel=max_parallel,
            skip_cache_miss=skip_cache_miss,
        ),
        {},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-samples",
        type=int,
        default=25,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./benchmark_results/",
        help="Output folder for results",
    )
    parser.add_argument(
        "--exp-spec",
        type=str,
        help="Experiment specification file",
    )
    parser.add_argument(
        "--list-exps",
        action="store_true",
        help="List available experiments and their parameters",
    )
    parser.add_argument(
        "--skip-cache-miss",
        action="store_true",
        help="Skip tasks that have not already been computed",
    )

    args = parser.parse_args()

    # List all available experiments
    if args.list_exps:
        print("Available experiments:")
        # Filter globals to only include functions that start with "experiment_"
        exp_funcs = [
            v
            for k, v in globals().items()
            if k.startswith("experiment_") and inspect.isfunction(v)
        ]

        # Print each experiment function name with a list of its parameters
        for func in exp_funcs:
            # Get the function signature
            signature = inspect.signature(func)
            # List of parameters excluding 'n_samples', 'output_folder', and 'skip_cache_miss'
            params = [
                name
                for name, param in signature.parameters.items()
                if name not in ["n_samples", "output_folder", "skip_cache_miss"]
            ]
            # Print the function name and its parameters
            print(f"\t{func.__name__}({', '.join(params)})")

        exit()

    # Run all experiments
    results = {}
    extra_infos = {}
    # ... load specifications
    with open(args.exp_spec, "r") as f:
        exp_spec = json.load(f)
    # ... run each experiment
    for exp in exp_spec:
        print(f"Running experiment: {exp['label']}")
        exp_label = exp["label"]
        # ... extract configuration
        config = {k: v for k, v in exp.items() if k != "method" and k != "label"}
        config["n_samples"] = args.n_samples
        config["output_folder"] = Path(args.output) / exp_label
        config["skip_cache_miss"] = args.skip_cache_miss
        print(f"\tConfig: {config}")
        # ... do it!
        function = globals().get(f"experiment_{exp['method']}")
        # ... process results
        res, extra_info = function(**config)
        if isinstance(res, list):
            results.update({f"{exp_label}_{k}": v for k, v in res})
        else:
            results[exp_label] = res
        extra_infos[exp_label] = extra_info

    # Gather all results that are missing (cache miss mentioned in error message)
    missing_results = {
        method: [
            res
            for res in results[method]
            if "error" in res and "cache miss" in res["error"].lower()
        ]
        for method in results
    }

    # Compile results into Pandas dataframe
    results_ = {
        "Task": [task for task in list(results.values())[0]],
    }
    results_.update(
        {
            method: [
                np.mean([res["score"] for res in results[task] if not "error" in res])
                for task in results
            ]
            for method, results in results.items()
        }
    )
    results = pd.DataFrame(results_).sort_values("Task").set_index("Task")
    del results_
    print(results)
    print("\n" * 2)
    print(f"Missing results: {missing_results}")


if __name__ == "__main__":
    main()
