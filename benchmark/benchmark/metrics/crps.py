# Taken from https://github.com/ServiceNow/regions-of-reliability/blob/main/ror/metrics/crps.py
"""
Copyright 2023 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np


def get_crps(
    targets: list,  # list of (num_timesteps, )
    forecasts: list,  # list of (num_timestes, num_samples)
):
    """
    Returns both GluonTS CRPS and series-wise-normalized CRPS
    """
    # Extract the forecasts and targets from the outputs
    series_quantile_losses = []
    series_target_abs_sums = []

    series_crps_list = []
    num_tokens_list = []

    for i in range(len(targets)):
        target = targets[i]
        forecast = forecasts[i]
        i_quantile_losses, i_target_abs_sum = crps_quantile(
            target,  # Of shape (num_timesteps, )
            forecast.transpose(),  # Of shape (num_samples, num_timesteps)
        )
        series_quantile_losses.append(i_quantile_losses)
        series_target_abs_sums.append(i_target_abs_sum)

        # Get the sum of the CRPS for the series by integrating the quantiles over [0, 1]
        crps_sum = i_quantile_losses.mean(axis=1).item()
        abs_sum = i_target_abs_sum.item()
        if abs_sum > 0:
            crps_normalized = crps_sum / abs_sum
            series_crps_list.append(crps_normalized)
            num_tokens_list.append(target.shape[0])
        else:
            series_crps_list.append(float("nan"))
            # Avoid changing the weighted average, since we will skip the NaNs in the numerator
            num_tokens_list.append(0)

    series_quantiles = np.concatenate(
        series_quantile_losses
    )  # Of shape (num_series, num_quantiles)
    # Set NumPy to ignore divide by zero errors
    with np.errstate(divide="ignore"):
        series_weighted_quantiles = series_quantiles / np.atleast_2d(
            np.array(series_target_abs_sums)
        )
    # series_crps_1 = series_weighted_quantiles.mean(axis=1)  # Of shape (num_series, )

    overall_target_abs_sum = np.concatenate(series_target_abs_sums).sum()
    overall_quantiles = series_quantiles.sum(axis=0)  # Of shape (num_quantiles, )
    # Set NumPy to ignore divide by zero errors and instead return inf
    with np.errstate(divide="ignore"):
        overall_weighted_quantiles = overall_quantiles / overall_target_abs_sum
    overall_gluonts_crps = overall_weighted_quantiles.mean()

    # Convert the list to np.array for ease of manipulation
    series_crps_2 = np.array(series_crps_list)
    num_tokens = np.array(num_tokens_list)

    # Weighted average by the number of tokens in each series, since we don't have
    overall_series_wise_normalized_crps = (
        np.nan_to_num(series_crps_2, 0) * num_tokens
    ).sum() / num_tokens.sum()

    return overall_gluonts_crps, overall_series_wise_normalized_crps


def gluonts_crps(
    targets: list,  # list of (num_timesteps, )
    forecasts: list,  # list of (num_timestes, num_samples)
):
    """
    Calculate the CRPS similarly to the GluonTS implementation.
    Calculates quantile loss for each series and then sums them up.
    Normalizes the quantile loss by the sum of the absolute values of the targets.
    Note that this means that the CRPS is not invariant to the scale of the targets:
    multiplying one target and its forecasts by the same constant will affect the overall CRPS.
    """
    # Extract the forecasts and targets from the outputs
    series_quantile_losses = []
    series_target_abs_sums = []
    for i in range(len(targets)):
        target = targets[i]
        forecast = forecasts[i]
        # target = outputs[i]["target"][np.newaxis, :]
        # forecast = np.expand_dims(outputs[i]["forecast"], 1).transpose((2, 1, 0))
        i_quantile_losses, i_target_abs_sum = crps_quantile(
            target,  # (num_timesteps, )
            forecast.transpose(),  # (num_samples, num_timesteps)
        )
        # crps_scores.append(crps_score)
        series_quantile_losses.append(i_quantile_losses)
        series_target_abs_sums.append(i_target_abs_sum)
    # average_crps_score = np.array(crps_scores).mean()

    series_quantiles = np.concatenate(
        series_quantile_losses
    )  # (num_series, num_quantiles)
    # Set NumPy to ignore divide by zero errors
    with np.errstate(divide="ignore"):
        series_weighted_quantiles = series_quantiles / np.atleast_2d(
            np.array(series_target_abs_sums)
        )
    series_crps = series_weighted_quantiles.mean(axis=1)  # (num_series, )

    overall_target_abs_sum = np.concatenate(series_target_abs_sums).sum()
    overall_quantiles = series_quantiles.sum(axis=0)  # (num_quantiles, )
    # Set NumPy to ignore divide by zero errors and instead return inf
    with np.errstate(divide="ignore"):
        overall_weighted_quantiles = overall_quantiles / overall_target_abs_sum
    overall_crps = overall_weighted_quantiles.mean()

    return overall_crps, series_crps


def series_wise_normalized_crps(
    targets: list,  # list of (num_timesteps, )
    forecasts: list,  # list of (num_timestes, num_samples)
):
    """
    Compute the CRPS and normalize the result independently for each series.
    The final result is the average CRPS, where the averaging is done per token,
    so series with more tokens will naturally have a higher weight.
    """
    series_crps_list = []
    num_tokens_list = []
    for target, forecast in zip(targets, forecasts):
        quantile_loss_sum, target_abs_sum = crps_quantile(target, forecast.transpose())
        # Get the sum of the CRPS for the series by integrating the quantiles over [0, 1]
        crps_sum = quantile_loss_sum.mean(axis=1).item()
        abs_sum = target_abs_sum.item()
        if abs_sum > 0:
            crps_normalized = crps_sum / abs_sum
            series_crps_list.append(crps_normalized)
            num_tokens_list.append(target.shape[0])
        else:
            series_crps_list.append(float("nan"))
            # Avoid changing the weighted average, since we will skip the NaNs in the numerator
            num_tokens_list.append(0)

    # Convert the list to np.array for ease of manipulation
    series_crps = np.array(series_crps_list)
    num_tokens = np.array(num_tokens_list)

    # Weighted average by the number of tokens in each series, since we don't have
    mean_crps = (np.nan_to_num(series_crps, 0) * num_tokens).sum() / num_tokens.sum()

    return mean_crps, series_crps


def crps_quantile(
    target: np.array,
    samples: np.array,
    quantiles: np.array = (np.arange(20) / 20.0)[1:],
) -> np.float32:
    # Compute the CRPS using the quantile scores
    # For a single series
    assert target.shape[0] == samples.shape[1]
    num_samples = samples.shape[0]

    sorted_samples = np.sort(samples, axis=0)

    quantile_scores = {}

    for q in quantiles:
        # From 0 to num_samples - 1 so that the 100% quantile is the last sample
        q_idx = int(np.round(q * (num_samples - 1)))
        q_value = sorted_samples[q_idx, :]

        # The absolute value is just there in case of numerical inaccuracies
        quantile_score = np.abs(2 * ((target <= q_value) - q) * (q_value - target))

        # result_sum += np.nanmean(quantile_score, axis=(0, 1))
        quantile_scores[q] = quantile_score.sum(axis=0)

    abs_target_sum = np.atleast_1d(np.abs(target).sum(axis=0))

    series_QuantileLoss = np.array(
        [quantile_scores[quantile] for quantile in quantiles]
    ).transpose()  # (num_series, num_quantiles)

    series_QuantileLoss = (
        series_QuantileLoss.reshape(1, -1)
        if series_QuantileLoss.ndim == 1
        else series_QuantileLoss
    )
    # if normalize:
    #     result_sum /= np.sum(np.abs(target))

    return series_QuantileLoss, abs_target_sum


def crps_slow(target: np.array, samples: np.array) -> np.float32:
    """This code is currently not adapted to our codebase."""
    # Compute the CRPS using the definition:
    # CRPS(y, X) = E |y - X| + 0.5 E |X - X'|, averaged over each dimension
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]
    num_samples = samples.shape[0]

    first_term = np.abs(samples - target[None, :, :]).mean(axis=0).mean()
    s = np.float32(0)
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            s += np.abs(samples[i] - samples[j]).mean()
    second_term = s / (num_samples * (num_samples - 1) / 2)

    return first_term - 0.5 * second_term
