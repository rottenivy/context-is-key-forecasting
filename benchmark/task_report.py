import base64
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from benchmark import ALL_TASKS
from datetime import datetime
from io import BytesIO


CONTEXT_PRETTY_NAMES = {
    "c_h": "History",
    "c_f": "Future",
    "c_i": "Intemporal",
    "c_cov": "Covariates",
    "c_causal": "Causal",
}


def _figure_to_html(figure):
    """
    Convert a matplotlib figure to an HTML string

    """
    # Save the figure to a byte array
    figure_bytes = BytesIO()
    figure.savefig(figure_bytes, format="png")
    plt.close(figure)
    figure_bytes.seek(0)

    # Convert byte array to base64 string
    figure_base64 = base64.b64encode(figure_bytes.read()).decode("utf-8")

    # Render as it would be in an HTML file
    figure_html = f'<img src="data:image/png;base64,{figure_base64}" class="img-fluid" alt="Figure"/>'

    return figure_html


def get_task_info(tasks):
    """
    Get a DataFrame with information about which tasks use which context flags

    """
    task_info = []
    for task_cls in tasks:
        tmp = {}
        tmp["task"] = task_cls.__name__
        tmp.update(task_cls().context_flags)
        task_info.append(tmp)

    return (
        pd.DataFrame(task_info).set_index("task").rename(columns=CONTEXT_PRETTY_NAMES)
    )


def plot_tasks_per_context_type(task_info):
    """
    Plot the number of tasks using each context type

    """
    # Count the number of tasks using each context type
    context_counts = task_info.sum()

    # Create a bar plot
    context_counts.plot(kind="bar", color="blue")
    plt.xlabel("Context Type")
    plt.ylabel("Number of Tasks")
    plt.title("Number of Tasks Using Each Context Type")

    plt.tight_layout()

    return plt.gcf()


def list_tasks_per_context_type(task_info):
    """
    Produce an HTML list where each context type is a heading and the tasks
    with a True value for that context type are listed under it as a bullet list

    """
    # Get the list of context types
    context_types = task_info.columns

    # Initialize the HTML string
    html = ""

    # Loop over the context types
    for context_type in context_types:
        # Add the context type as a heading
        html += f"<h3 class='text-primary'>{context_type}</h3>\n"

        # Get the tasks using the context type
        tasks = task_info[task_info[context_type]].index

        if len(tasks) == 0:
            html += "<p>No tasks use this context type</p>\n"
        else:
            # Add the tasks as a bullet list
            html += "<ul class='list-group'>\n"
            for task in tasks:
                html += f"<li class='list-group-item'><a href='./{task}.html'>{task}</a></li>\n"
            html += "</ul>\n"

    return html


def create_task_summary_page(tasks):
    """
    Create a summary page for the benchmark tasks

    """
    for task in tasks:
        with open(f"{task.name}.html", "w") as f:
            f.write("Hello world!")


def task_info_heatmap(task_info):
    """
    Render the task info matrix as a heatmap
    """
    # Convert boolean values to integers
    task_info_int = task_info.astype(int)

    # Create the heatmap
    plt.figure()
    ax = sns.heatmap(
        task_info_int,
        cmap=sns.color_palette(["white", "red"]),
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        yticklabels=True,
        xticklabels=True,
    )

    # Add a border around the heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("black")

    # Rotate the x-tick labels
    plt.xticks(rotation=90)

    plt.title("Context Type by Task")
    plt.xlabel("Context Types")
    plt.ylabel("Tasks")

    plt.tight_layout()

    return plt.gcf()


if __name__ == "__main__":
    task_info = get_task_info(ALL_TASKS)
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_by_context_bar = _figure_to_html(plot_tasks_per_context_type(task_info))
    task_info_heatmap = _figure_to_html(task_info_heatmap(task_info))
    task_by_context_list = list_tasks_per_context_type(task_info)

    create_task_summary_page(ALL_TASKS)

    report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Task Overview</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            background-color: #f8f9fa;
        }}
        .container {{
            margin-top: 20px;
        }}
        .section {{
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 0.25rem;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .section h2 {{
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        .list-group-item {{
            background-color: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="section">
            <h1 class="text-center text-primary">Benchmark Task Overview</h1>
            <p class="text-center">Generated on {generation_time}</p>
        </div>

        <div class="section">
            <h2 class="text-primary">Tasks by Context Type</h2>
            <div class="mb-4">
                There is a total of {len(task_info)} tasks in the benchmark.
            </div>

            {task_by_context_list}
        </div>

        <div class="section">
            <h2 class="text-primary">Visualizations</h2>
            <div class="mb-4">
                {task_by_context_bar}
            </div>
            <div class="mb-4">
                {task_info_heatmap}
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

    with open("index.html", "w") as f:
        f.write(report)
