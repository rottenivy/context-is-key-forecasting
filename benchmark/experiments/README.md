# Experiments

This readme gives an overview of the main job launching components.

## run_baselines.py

This is a command line tool to run experiments with baselines and produce reports.

#### Experimenting with a baseline
* This command line tools executes experiments specified in json files.
* The Python file specifies experiment functions, each named with `experiment_<method>`, where `<method>` is the method name to be used in json files.
* Methods can take arguments, which are also specified in json.
* Here's an example that first runs GPT-4o mini (without context) and then Lag-Llama (sequentially).
```
[
    {"label": "GPT-4o-mini (no ctx)", "method": "gpt", "llm": "gpt-4o-mini", "use_context": false},
    {"label": "Lag-Llama", "method": "lag_llama"}
]
```
* Such a file can be passed to the command via the exp-spec argument, i.e., `python run_baselines.py --exp-spec myfile.json`
* The command line tool also allows to specify the output directory (for plots) and the number of samples to be drawn from the models.

#### Summary table
To generate a summary table with the results of multiple methods, simply make a json file with all the experiments you care about and call the command line tool. It will iterate through every single task/seed combination for all experiments (should be fast due to caching) and produce the final table. Note that you can add the `--skip-cache-miss` argument to skip any result that isn't currently computed (e.g., GPT failed due to some error but you don't want to re-query it right now).


## exp_launcher.py

This is a script that lists all experiments (json) in a directory (--expdir argument; default `./experiments`) and launches them on toolkit.

* The json files are expected to be suffixed with either `_c<x>` or `_g<x>`, which respectively specify that the job should be run on x CPUs or GPUs (e.g., _g3 means 3 x GPUs).
* The script will launch individual jobs on toolkit for each of the experiments and they will run in parallel.
* You can use `python exp_launcher.py --killall` to kill all your running experiments (this will only kill what was launched by this script).