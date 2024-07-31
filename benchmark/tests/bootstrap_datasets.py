from benchmark import ALL_TASKS

# TODO: Remove this when dominick data loader is merged
from benchmark.predictable_grocer_shocks import __TASKS__ as DOMINICK_TASKS

ALL_TASKS = list(set(ALL_TASKS) - set(DOMINICK_TASKS))

print("Downloading datasets for all tasks...")
for task in ALL_TASKS:
    # Create a random instance of the task
    # This will force the download of any required dataset
    task()
print("Done.")
