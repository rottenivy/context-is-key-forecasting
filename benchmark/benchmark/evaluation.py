from collections import defaultdict

from . import ALL_TASKS


def evaluate_all_tasks(method_callable, seeds=5, n_samples=50):
    """
    Evaluates a method on all tasks for a number of seeds and samples

    Parameters:
    -----------
    method_callable: callable
        A callable that receives a task instance and returns a prediction samples.
        The callable should expect the following kwargs: task_instance, n_samples
    seeds: int
        Number of seeds to evaluate the method
    n_samples: int
        Number of samples to generate for each prediction

    Returns:
    --------
    results: dict
        A dictionary with the results of the evaluation.
        Keys are task names and values are lists of dictionaries
        with metrics and relevant information.

    """
    results = defaultdict(list)
    for task_cls in ALL_TASKS:
        for seed in range(1, seeds + 1):
            task = task_cls(seed=seed)

            results[task_cls.__name__].append(
                {
                    "seed": seed,
                    "score": task.evaluate(
                        method_callable(task_instance=task, n_samples=n_samples)
                    ),
                }
            )

    return results
