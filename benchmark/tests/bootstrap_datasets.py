from benchmark import ALL_TASKS

# TODO: Remove this when dominick data loader is merged
from benchmark.predictable_grocer_shocks import __TASKS__ as DOMINICK_TASKS

for task in DOMINICK_TASKS:
    if task in ALL_TASKS:
        ALL_TASKS.remove(task)

print("Downloading datasets for all tasks...")
for task in ALL_TASKS:
    # Create a random instance of the task
    # This will force the download of any required dataset
    task()
print("Done.")
