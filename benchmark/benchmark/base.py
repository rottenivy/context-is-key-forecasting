"""
Base classes for the benchmark

"""

import numpy as np
import pandas as pd
from termcolor import colored
import scipy
import statsmodels.tsa.tsatools

from abc import ABC, abstractmethod

from .metrics.crps import crps_quantile

import networkx as nx
import matplotlib.pyplot as plt


class BaseTask(ABC):
    """
    Base class for a task

    Parameters:
    -----------
    seed: int
        Seed for the random number generator
    fixed_config: dict
        Fixed configuration for the task

    """

    def __init__(self, seed: int = None, fixed_config: dict = None):
        self.random = np.random.RandomState(seed)

        # Instantiate task parameters
        if fixed_config is not None:
            self.past_time = fixed_config["past_time"]
            self.future_time = fixed_config["future_time"]
            self.constraints = fixed_config["constraints"]
            self.background = fixed_config["background"]
            self.scenario = fixed_config["scenario"]
        else:
            self.constraints = None
            self.background = None
            self.scenario = None
            self.random_instance()

        config_errors = self.verify_config()
        if config_errors:
            raise RuntimeError(
                f"Incorrect config for {self.__class__.__name__}: {config_errors}"
            )

    @property
    def name(self) -> str:
        """
        Give the name of the task, for reporting purpose

        Returns:
        --------
        name: str
            The name of the task
        """
        return self.__class__.__name__

    @property
    def seasonal_period(self) -> int:
        """
        This returns the period which should be used by statistical models for this task.
        If negative, this means that the data either has no period, or the history is shorter than the period.
        """
        # By default, uses the frequency of the data to guess the period.
        # This should be overriden for tasks for which this guess fails.
        freq = self.past_time.index.freq
        if not freq:
            freq = pd.infer_freq(self.past_time.index)
        period = statsmodels.tsa.tsatools.freq_to_period(freq)
        return period

    def verify_config(self) -> list[str]:
        """
        Check whether the task satisfy the correct format for its parameters.

        Returns:
        --------
        errors: list[str]
            A list of textual descriptions of all errors in the format
        """
        errors = []
        # A few tests to make sure that all tasks use a compatible format for the parameters
        # Note: Only the parameters which are used elsewhere are current tested
        if not isinstance(self.past_time, pd.DataFrame):
            errors.append(
                f"past_time is not a pd.DataFrame, but a {self.past_time.__class__.__name__}"
            )
        if not isinstance(self.future_time, pd.DataFrame):
            errors.append(
                f"future_time is not a pd.DataFrame, but a {self.future_time.__class__.__name__}"
            )
        return errors

    @abstractmethod
    def evaluate(self, samples: np.ndarray):
        """
        Evaluate success based on samples from the inferred distribution

        Parameters:
        -----------
        samples: np.ndarray, shape (n_samples, n_time, n_dim)
            Samples from the inferred distribution

        Returns:
        --------
        metric: float
            Metric value

        """
        pass

    @abstractmethod
    def random_instance(self):
        """
        Generate a random instance of the task and instantiate its data

        """
        pass


class UnivariateCRPSTask(BaseTask):
    """
    A base class for tasks that require forecasting a single series and that use CRPS for evaluation
    We use the last column of `future_time` as the ground truth for evaluation
    """

    def evaluate(self, samples):
        if len(samples.shape) == 3:
            samples = samples[:, :, 0]

        # This is the dual of pd.Series.to_frame(), compatible with any series name
        only_column = self.future_time.columns[-1]
        target = self.future_time[only_column]
        return crps_quantile(target=target, samples=samples)[0].mean()


class CausalUnivariateCRPSTask(BaseTask):
    """
    Base class for all synthetic causal tasks that require forecasting a single series and that use CRPS for evaluation

    """

    # To be used only for the instantaneous graph
    def generate_dag(self, num_nodes, degree):
        """
        Generate Erdos-Renyi random graph with a given degree

        Required only for making the instantaneous graphs since the lagged graphs
        do not have to be acyclic.
        """

        total_edges = num_nodes * (num_nodes - 1) // 2
        total_expected_edges = degree * num_nodes

        if total_expected_edges >= total_edges:
            print(
                colored(
                    f"Warning: For d={num_nodes} nodes and degree={degree}, full graphs will be generated",
                    "red",
                )
            )

        p_threshold = float(total_expected_edges) / total_edges
        p_edge = (self.random.rand(num_nodes, num_nodes) < p_threshold).astype(float)
        L = np.tril(p_edge, k=-1)

        P = self.random.permutation(np.eye(num_nodes, num_nodes))
        G = P.T @ L @ P
        return G

    def check_dagness(self, inst_graph):
        num_nodes = inst_graph.shape[0]
        assert (
            np.trace(scipy.linalg.expm(inst_graph * inst_graph)) - num_nodes == 0
        ), colored(
            "Error in DAG generation: Instantaneous graph is not acyclic!", "red"
        )

    def random_graph(
        self, num_nodes, intra_degree, inter_degree, lag, instantaneous_edges, split
    ):
        """
        A function to generate a random (instantaneous, lagged) DAG graph for the causal task

        Parameters
        ----------
            num_nodes: int

            intra_degree: float
                Expected edges for instantaneous graph

            inter_degree: float
                Expected edges for lagged graph

            lag: int
                Lag timesteps for the causal

            instantaneous_edges: bool
                If True, the graph will have instantaneous connections and intra_degree will be used
                Else intra_degree is not used for any step of the random DAG generation.

            split: Tuple(List, List)
                Forecast variable list and covariate list

        Returns
        -------
        A complete DAG such that the forecast variable has at least one covariate as a parent.

            graph: np.array
                Binary adjacency matrix of shape (lag+1, d, d)
        """

        assert lag > 0, "Lag must be greater than 0"
        assert num_nodes > 1, "DAG must have at least 2 nodes"
        assert not instantaneous_edges, "Only lagged graphs are supported for now"

        forecast_vars, covariates = split

        if instantaneous_edges:
            inst_G = self.generate_dag(num_nodes, intra_degree)
        else:
            inst_G = np.zeros((num_nodes, num_nodes))

        self.check_dagness(inst_G)

        lagged_G = np.zeros((lag, num_nodes, num_nodes))

        """
            expected_lag_edges = inter_degree * num_nodes
            total_lag_edges = num_nodes * num_nodes * lag

            edge_probs = expected_lag_edges / total_lag_edges
        """
        edge_probs = inter_degree / (num_nodes * lag)

        total_edges_per_lag_graph = num_nodes**2
        expected_edges_per_lag_graph = inter_degree * num_nodes
        edge_probs = float(expected_edges_per_lag_graph) / total_edges_per_lag_graph

        for l in range(lag):
            lagged_G[l, :, :] = (
                self.random.uniform(size=(num_nodes, num_nodes)) < edge_probs
            )

        # Make sure there is at least one historical parent
        zero_historical_parent_nodes = np.where(np.sum(lagged_G, axis=(0, 1)) == 0)[0]
        if len(zero_historical_parent_nodes) > 0:
            lag_idxs = self.random.choice(lag, len(zero_historical_parent_nodes))
            random_parent_idxs = self.random.choice(
                num_nodes, len(zero_historical_parent_nodes)
            )
            lagged_G[lag_idxs, random_parent_idxs, zero_historical_parent_nodes] = 1

        # Ensure each forecast variable has a historical parent in the covariate set for every lag step
        for l in range(lag):
            # Get all connections from covariate_{t-l} to forecast_{t}
            lag_submatrix = lagged_G[l][
                np.ix_(covariates, forecast_vars)
            ]  # (len(covariates), len(forecast_vars))

            # Check if there is at least one connection from covariate_{t-l} to forecast_{t}
            has_covariate_parent = lag_submatrix.sum(axis=0) > 0.0

            # For any forecast variable without a covariate parent, add a random covariate parent
            for i, node in enumerate(forecast_vars):
                if not has_covariate_parent[i]:
                    random_parent = self.random.choice(covariates)
                    lagged_G[l, random_parent, node] = 1

        complete_graph = np.concatenate((inst_G[None], lagged_G), axis=0)
        return complete_graph

    def plot_temporal_graph(self, complete_graph):
        """
        Function to visualize the instantaneous and lagged graphs, to aid in debugging
        """
        L = complete_graph.shape[0] - 1
        d = complete_graph.shape[1]
        G = nx.DiGraph()

        for lag in range(L + 1):
            for i in range(d):
                G.add_node((lag, i), label=f"{i}")

        # Add edges based on the adjacency matrices
        for lag in range(L + 1):
            for i in range(d):
                for j in range(d):
                    if complete_graph[lag, i, j] == 1:
                        G.add_edge(
                            (L - lag, i), (L, j)
                        )  # Connection from (t-lag, variable i) to (t, variable j)

        pos = {}
        for lag in range(L + 1):
            for i in range(d):
                pos[(lag, i)] = (lag, -i)

        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels={(lag, i): f"{i}" for lag in range(L + 1) for i in range(d)},
            node_size=500,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
            ax=ax,
        )

        # Set labels for each column
        timestep_labels = [f"T-{L-lag}" if lag != L else "T" for lag in range(L + 1)]
        for lag, label in enumerate(timestep_labels):
            ax.text(
                lag,
                -d + 0.3,
                label,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
                fontweight="bold",
                color="green",
            )

        ax.set_title("Temporal Adjacency Matrix Visualization")
        plt.show()
        plt.savefig("temporal_graph.png")

    def evaluate(self, samples):
        if len(samples.shape) == 3:
            samples = samples[:, :, 0]

        # This is the dual of pd.Series.to_frame(), compatible with any series name
        only_column = self.future_time.columns[0]
        target = self.future_time[only_column]
        return crps_quantile(target=target, samples=samples)[0].sum()

    def node_i_parent_descriptions(self, W, L, node, desc):
        for l in range(1, L + 1):
            parents = W[l, :, node]
            # Get index
            parents = np.where(parents != 0)[0]
            if len(parents) == 0:
                return f"No parents for variable {node} at lag {l}"

            elif desc == "minimal":
                parent_vars = []
                for parent in parents:
                    parent_vars.append(f"X_{parent}")
                return f"Parents for variable X_{node} at lag {l}: {parent_vars}"

            elif desc == "edge_weights":
                parent_vars = []
                coeff_parent_vars = []
                for parent in parents:
                    coefficient = W[l, parent, node]
                    coeff_parent_vars.append(f"{coefficient} * X_{parent}")
                    parent_vars.append(f"X_{parent}")

                expression = " + ".join(coeff_parent_vars)
                return f"Parents for variable X_{node} at lag {l}: {parent_vars} affect the forecast variable as {expression}"
