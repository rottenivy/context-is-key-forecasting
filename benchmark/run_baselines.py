"""
Run all baselines on all tasks and save the results to a Pandas dataframe.

"""

import logging
import numpy as np
import pandas as pd

from benchmark.baselines.gpt_processes import GPTForecaster
from benchmark.baselines.lag_llama import lag_llama
from benchmark.baselines.llm_processes import LLMPForecaster
from benchmark.baselines.naive import oracle_baseline, random_baseline
from benchmark.evaluation import evaluate_all_tasks


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    n_samples = 50

    # To plot the results, add: output_folder="./figures/baseline_name/" as an argument to evaulate_all_tasks
    results = {}

    results["random"] = evaluate_all_tasks(
        random_baseline,
        n_samples=n_samples,
        output_folder="./benchmark_results/random/",
    )
    results["oracle"] = evaluate_all_tasks(
        oracle_baseline,
        n_samples=n_samples,
        output_folder="./benchmark_results/oracle/",
    )
    # results["lag_llama"] = evaluate_all_tasks(lag_llama, n_samples=n_samples, output_folder="./benchmark_results/lag_llama/")

    # # OpenAI baselines
    # openai_costs = {
    #     "gpt-4o": {"input": 0.005, "output": 0.015},
    #     "gpt-35-turbo": {"input": 0.002, "output": 0.002}
    # }
    # open_ai_cost = 0
    # for llm in ["gpt-4o", "gpt-35-turbo"]:
    #     for include_context in [True, False]:
    #         gpt_forecaster = GPTForecaster(model=llm, use_context=include_context, token_cost=openai_costs[llm])
    #         results[f"gpt_{llm}_{'ctx' if include_context else 'no_ctx'}"] = evaluate_all_tasks(
    #             gpt_forecaster,
    #             n_samples=n_samples,
    #             output_folder=f"./benchmark_results/{gpt_forecaster.cache_name}",
    #         )
    #         open_ai_cost += gpt_forecaster.total_cost
    #         print(open_ai_cost)
    #         del gpt_forecaster

    # # LLMP baselines
    # for llm in ["llama-3-8B", "phi-3-mini-128k-instruct"]:
    #     for include_context in [True, False]:
    #         llmp_forecaster = LLMPForecaster(llm_type=llm, include_context=include_context)
    #         results[f"llmp_{llm}_{'ctx' if include_context else 'no_ctx'}"] = evaluate_all_tasks(
    #             llmp_forecaster,
    #             n_samples=n_samples,
    #             output_folder=f"./benchmark_results/{llmp_forecaster.cache_name}",
    #         )

    # Compile results into Pandas dataframe
    results_ = {
        "Task": [task for task in list(results.values())[0]],
    }
    results_.update(
        {
            method: [
                np.mean([res["score"] for res in results[task]]) for task in results
            ]
            for method, results in results.items()
        }
    )
    results = pd.DataFrame(results_)
    del results_

    # Print results and save them
    print(results)
    results.to_csv("./benchmark_results/summary.csv")
