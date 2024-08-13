from .base import CausalUnivariateCRPSTask
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from termcolor import colored
from abc import abstractmethod

"""
    Usage:
    task = BivariateCategoricalLinSVAR(seed=1)
    plot_forecast_with_covariates(task, "causal.png")
"""


class BivariateCategoricalLinSVARBaseTask(CausalUnivariateCRPSTask):
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
            "lag": 3,
            "instantaneous_edges": False,
            "burn_in": 30,
            "noise_type": "gauss",
            "noise_scale": 0.1,
            "time_window": 200,
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

    def generate_regimes(self, T, values, value_type, double_regimes, verbalize=True):
        array = []
        regime_lengths = []
        regime_values = []
        pred_time_covariate_desc = []
        remaining_length = T
        assert value_type in ["fluctuate", "const"]

        time_elapsed = 0
        ha = 0
        while remaining_length > 0:
            if value_type == "fluctuate":
                regime_length = self.random.randint(low=10, high=20)
                regime_length = min(
                    regime_length, remaining_length
                )  # Ensure it doesn't exceed remaining length
            else:
                regime_length = remaining_length

            regime_value = self.random.choice(values)
            ha += 1
            # Halfway point
            if double_regimes:
                print(ha, regime_length, time_elapsed)
                if (
                    remaining_length > T // 2
                    and remaining_length - regime_length <= T // 2
                ):
                    print("Right")
                    first_half = T // 2 - time_elapsed
                    second_half = regime_length - first_half

                    regime_values.append(regime_value)
                    regime_values.append(regime_value * 2)

                    array.extend([regime_value] * first_half)
                    array.extend([regime_value * 2] * second_half)

                    regime_lengths.append(first_half)
                    regime_lengths.append(second_half)

                elif remaining_length <= T // 2:
                    regime_values.append(regime_value * 2)
                    regime_lengths.append(regime_length)
                    array.extend([regime_value * 2] * regime_length)

                else:
                    regime_values.append(regime_value)
                    regime_lengths.append(regime_length)
                    array.extend([regime_value] * regime_length)
            else:
                regime_lengths.append(regime_length)
                array.extend([regime_value] * regime_length)

            time_elapsed += regime_length
            remaining_length -= regime_length

        # if double_regimes:
        #     import pdb; pdb.set_trace()

        if verbalize:
            for i in range(len(regime_values)):
                text = f"{regime_values[i]} for {regime_lengths[i]} timesteps"
                pred_time_covariate_desc.append(text)

        return np.array(array), pred_time_covariate_desc

    def generate_time_series(
        self, W, history_length, n_samples, noise_type="gauss", noise_scale=0.1
    ):
        d = self.causal_config["num_nodes"]
        L = self.causal_config["lag"]
        total_length = self.causal_config["burn_in"] + 2 * n_samples
        pred_length = n_samples - history_length

        historical_covariates, hist_cov_desc = self.generate_regimes(
            T=total_length - n_samples + history_length,
            values=[2, 6, 8],
            value_type="const",
            double_regimes=False,
        )
        future_covariates, pred_cov_desc = self.generate_regimes(
            T=pred_length,
            values=[10, 12, 30],
            value_type="fluctuate",
            double_regimes=True,
        )

        # future_covariates[-pred_length // 2 :] *= 2
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

        X_post_burn_in = X[-n_samples:, :]
        return X_post_burn_in, historical_covariates[0], (hist_cov_desc, pred_cov_desc)

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
        assert len(covariate_idxs) == 1  # Bivariate setting has only 1 `covariate`

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

                X_post_burn_in, const_hist_value, cov_desc = self.generate_time_series(
                    W, history_length, n_samples, noise_type, noise_scale
                )

                print(cov_desc)

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
        self.past_time = full_time_series_df[:history_length]
        self.future_time = full_time_series_df[history_length:]

        self.graph = W
        self.historical_parents = self.get_historical_parents(full_graph)

        # Set scenario, constraints and background
        background = f"The variable to forecast as well as the covariate, all are generated from a linear Structural Vector Autoregressive (SVAR) model with additive {noise_type} noise and a noise scale of {noise_scale},\n"
        background += f"with lag = {L}.\nVariable to forecast: X_{forecast_variable}."
        self.background = background

        self.scenario = self.get_scenario(
            const_hist_value, history_length, pred_length, cov_desc
        )
        self.constraints = None

        self.causal_context = self.get_causal_context(W, L)
        print(self.causal_context)

    def get_scenario(self, const_hist_value, history_length, pred_length, cov_desc):
        _, pred_cov_desc_list = cov_desc
        pred_cov_desc = ", ".join(pred_cov_desc_list)

        line1 = f"The task is to forecast the value of the variable X_{self.causal_config['num_nodes'] - 1} at time t, given the values of the covariate X_0 and the variable X_{self.causal_config['num_nodes'] - 1} itself at times t-1, ... t-{self.causal_config['lag']}."
        line2 = f"For the first {history_length} time steps, the covariate X_0 is constant at {const_hist_value}."
        line3 = f"For the next {pred_length} time steps, the covariate X_0 takes a value of {pred_cov_desc}."

        scenario = f"{line1}\n{line2}\n{line3}"
        return scenario

    @abstractmethod
    def get_causal_context(self, W, lag):
        pass


# Causal Context Level 1
class MinimalCausalContextBivarCategoricalLinSVAR(BivariateCategoricalLinSVARBaseTask):
    def get_causal_context(self, W, lag):
        d = W.shape[-1]
        graph_desc = []
        for i in range(d):
            desc_i = self.node_i_parent_descriptions(W, lag, i, desc="minimal")
            graph_desc.append(desc_i)
            # graph_desc.append("-----------")

        textual_causal_desc = "\n".join(graph_desc)
        causal_context = f"The causal parents affect the child variables at different lags. There are no instantaneous edges in the graph.\n"
        causal_context += f"The causal parents for each variable is given below:\n{textual_causal_desc}\n"
        return causal_context


# Causal Context Level 2
class FullCausalContextBivarCategoricalLinSVAR(BivariateCategoricalLinSVARBaseTask):
    def get_causal_context(self, W, lag):
        d = W.shape[-1]
        graph_desc = []
        for i in range(d):
            desc_i = self.node_i_parent_descriptions(W, lag, i, desc="edge_weights")
            graph_desc.append(desc_i)
            # graph_desc.append("-----------")

        textual_causal_desc = "\n".join(graph_desc)
        causal_context = f"The causal parents affect the child variables at different lags. There are no instantaneous edges in the graph.\n"
        causal_context += f"The causal parents for each variable is given below:\n{textual_causal_desc}.\n"
        return causal_context


__TASKS__ = [
    MinimalCausalContextBivarCategoricalLinSVAR,
    FullCausalContextBivarCategoricalLinSVAR,
]
