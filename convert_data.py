from cik_benchmark.tasks.general_traffic_tasks import TrafficTask_Random


if __name__ == '__main__':
    task = TrafficTask_Random(fresh_data=False)
    task.save_dataset_to_arrow(path="data/traffic_dataset.arrow")
