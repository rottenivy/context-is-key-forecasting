from .base import CausalUnivariateCRPSTask
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from termcolor import colored

"""
    Usage:
    task = BivariateCategoricalLinSVAR(seed=1)
    plot_forecast_with_covariates(task, "causal.png")
"""


class BivariateCategoricalLinSVAR(CausalUnivariateCRPSTask):
    """
    A task where there are two variables X and Y
    X_t depends linearly on X_{t-1:t-L} and Y_{t-1:t-L} + gaussian noise
    Y are categorical variables and undergo distributional shifts at pred time

    E.g., Y =
    [00101020001010202... | 113435343534433534...]
    [hist.. | pred...]
    """

    def __init__(self, fixed_config: dict = None, seed: int = None):

        self.causal_config = {
            "num_nodes": 2,
            "lag": 1,
            "instantaneous_edges": False,
            "burn_in": 30,
            "noise_type": "gauss",
            "noise_scale": 0.1,
            "time_window": 1000,
            "max_data_gen_trials": 100,
            "num_forecast_vars": 1,
        }
        super().__init__(seed=seed, fixed_config=fixed_config)

    def init_weights(self, full_graph):
        """
        Initialize the weighted adjacency matrix for the linear model
        """
        # Sample weights from [-2., 0.5] U [0.5, 2.]
        weights = self.random.uniform(0.5, 2, size=full_graph.shape)
        weights[
            self.random.rand(*weights.shape) < 0.5
        ] *= -1  # Half of the weights are negative

        weighted_adjacency_matrix = full_graph * weights
        return weighted_adjacency_matrix

    def generate_time_series(
        self, W, history_length, n_samples, noise_type="gauss", noise_scale=0.1
    ):
        d = self.causal_config["num_nodes"]
        L = self.causal_config["lag"]
        total_length = self.causal_config["burn_in"] + 2 * n_samples
        pred_length = n_samples - history_length

        historical_covariates = self.random.choice(
            4, size=(total_length - n_samples + history_length,)
        )
        future_covariates = self.random.choice([7, 8, 9], size=(pred_length,))
        future_covariates[-pred_length // 2 :] *= 2
        covariate_values = np.concatenate(
            [historical_covariates, future_covariates]
        )  # (total_length, )

        inst_W, lagged_W = W[0], W[1:]
        lagged_W = lagged_W.reshape(L * d, d)
        assert inst_W.shape == (d, d)

        X = np.zeros([total_length, d])
        # idx 0 is covariate and idx 1 is causal variable
        X[:, 0] = covariate_values

        Xlags = np.zeros([total_length, L, d])
        Xlags = Xlags.reshape(total_length, L * d)

        g_intra = nx.DiGraph(inst_W)
        g_inter = nx.bipartite.from_biadjacency_matrix(
            csr_matrix(lagged_W), create_using=nx.DiGraph
        )

        ordered_vertices = list(nx.topological_sort(g_intra))

        for t in range(total_length):
            for j in ordered_vertices:
                if j in [0]:
                    pass

                else:
                    parents = list(g_intra.predecessors(j))
                    parents_prev = list(g_inter.predecessors(j + L * d))

                    X[t, j] = (
                        +X[t, parents] @ inst_W[parents, j]
                        + Xlags[t, parents_prev] @ lagged_W[parents_prev, j]
                    )

                    if noise_type == "gauss":
                        X[t, j] = X[t, j] + self.random.normal(scale=noise_scale)
                    elif noise_type == "exp":
                        X[t, j] = X[t, j] + self.random.exponential(scale=noise_scale)
                    elif noise_type == "gumbel":
                        X[t, j] = X[t, j] + self.random.gumbel(scale=noise_scale)

            if (t + 1) < total_length:
                Xlags[t + 1, :] = np.concatenate([X[t, :], Xlags[t, :]])[: d * L]

        assert np.allclose(Xlags[1:], X[:-1], equal_nan=True)
        X_post_burn_in = X[-n_samples:, :]
        return X_post_burn_in

    def get_historical_parents(self, full_graph):
        num_nodes = full_graph.shape[-1]
        # For each variable (column), tells which nodes are parents aggregated across the lag timesteps
        historical_parent_matrix = np.sum(full_graph[1:], axis=0) > 0

        historical_parents = []
        for i in range(num_nodes):
            parents = np.where(historical_parent_matrix[:, i])[0]
            historical_parents.append(parents.tolist())

        assert len(historical_parents) == num_nodes
        return historical_parents

    def random_instance(self):
        """
        Generate a random instance of the task and instantiate its data
        """
        d = self.causal_config["num_nodes"]
        L = self.causal_config["lag"]
        max_data_gen_trials = self.causal_config["max_data_gen_trials"]
        n_samples = self.causal_config["time_window"]
        noise_type = self.causal_config["noise_type"]
        noise_scale = self.causal_config["noise_scale"]
        num_forecast_vars = 1

        attempt = 0
        simulate_flag = True

        X_post_burn_in = None
        full_graph = None
        W = None
        historical_parents = None

        vars_ = set(np.arange(d).tolist())

        # Only 1 variable to forecast, and it has to be the last variable in the df
        # for CRPS Evaluation: Hardcoded for now
        forecast_variable = d - 1
        covariate_idxs = vars_ - set([forecast_variable])

        full_graph = np.zeros((L + 1, d, d))
        full_graph[1:, :, forecast_variable] = 1

        history_length = int(0.8 * n_samples)
        pred_length = n_samples - history_length

        while simulate_flag:
            attempt += 1
            max_data_gen_trials -= 1
            if max_data_gen_trials < 0:
                simulate_flag = False
                raise ValueError(
                    f"Could not generate data after {max_data_gen_trials} trials"
                )

            try:
                W = self.init_weights(full_graph)

                X_post_burn_in = self.generate_time_series(
                    W, history_length, n_samples, noise_type, noise_scale
                )

                # Check if X_post_burn_in has NaNs
                if np.isnan(X_post_burn_in).any():
                    print(
                        colored(
                            f"Has NaN, continuing.... (attempt {attempt} over)", "blue"
                        )
                    )
                    continue

            except (OverflowError, FloatingPointError):
                print(
                    colored(
                        f"Has OverflowError or FloatingPointError.... (attempt {attempt} over)",
                        "blue",
                    )
                )
                continue

            # Find the max of X_post_burn_in
            if np.max(np.abs(X_post_burn_in)) < 1e4 or max_data_gen_trials <= 0:
                simulate_flag = False

            else:
                print(
                    colored(
                        f"Has max value greater than 1e4, continuing.... (attempt {attempt} over)",
                        "blue",
                    )
                )

        if max_data_gen_trials <= 0:
            raise ValueError(
                f"Could not simulate data, consider reducing noise, time window, or weight magnitudes"
            )

        # Split X_post_burn_in into history and future
        full_time_series_df = pd.DataFrame(X_post_burn_in)

        history_df = full_time_series_df[:history_length]
        future_df = full_time_series_df[history_length:]

        self.past_time = history_df[forecast_variable].to_frame()
        self.future_time = future_df[forecast_variable].to_frame()

        self.past_covariates = history_df.drop(columns=[forecast_variable])
        self.future_covariates = future_df.drop(columns=[forecast_variable])

        self.graph = W
        self.historical_parents = self.get_historical_parents(full_graph)

        # Set c_causal
        graph_desc = []
        for i in range(d):
            desc_i = self.node_i_parent_descriptions(W, L, i)
            graph_desc.append(desc_i)
            graph_desc.append("-----------")

        textual_causal_desc = "\n".join(graph_desc)
        self.context_causal = f"The causal parents affect the causal child at different lags. The causal parents for each variable is given below:\n"
        self.context_causal += f"{textual_causal_desc}\n"

        self.context_causal += (
            f"There are no instantaneous (or contemporaneous) edges in the graph."
        )

        # Set scenario, constraints and background
        scenario = f"The variable to forecast as well as the covariate, all are generated from a linear Structural Vector Autoregressive (SVAR) model with {noise_type} noise and a noise scale of {noise_scale}.\n"
        scenario += f"Total number of variables: {d}, lag: {L}. Variables to forecast: {[forecast_variable]}."
        self.scenario = scenario

        self.constraints = None
        self.background = None

        print(self.context_causal)


__TASKS__ = [BivariateCategoricalLinSVAR]
