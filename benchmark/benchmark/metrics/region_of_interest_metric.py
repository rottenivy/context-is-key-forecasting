from pandas import Timestamp
import numpy as np


class RegionOfInterestMetric:
    """
    A class to calculate a given metric on a region of interest.
    """

    def __init__(self, metric, target, region_of_interest=None, roi_weights=None):
        self.metric = metric
        self.target = target
        if region_of_interest is not None:
            self.region_of_interest = self.get_region_of_interest_mask(
                region_of_interest
            )
        self.roi_weights = self.get_target_weights(roi_weights)

    def get_region_of_interest_mask(self, region_of_interest, by="mask"):
        """ """
        if by == "mask":
            if not all(isinstance(x, bool) for x in region_of_interest):
                raise ValueError("Mask must be a boolean array")
            if len(region_of_interest) != len(self.target):
                raise ValueError("Mask length does not match target length.")
            return region_of_interest
        elif by == "indices":
            return self.get_region_of_interest_by_indices(region_of_interest)
        elif by == "timestamps":
            return self.get_region_of_interest_by_timestamps(region_of_interest)

    def get_region_of_interest_by_indices(self, indices):
        # check the dtype
        assert all(isinstance(x, int) for x in indices)
        region_of_interest_mask = np.zeros(len(self.target), dtype=bool)
        # if dtype of list elements is int, set region of interest by indices
        region_of_interest_mask[indices] = True

        return region_of_interest_mask

    def get_region_of_interest_by_timestamps(self, timestamps):
        # check the dtype
        assert all(isinstance(x, Timestamp) for x in timestamps)
        region_of_interest_mask = np.zeros(len(self.target), dtype=bool)
        # if dtype of list elements is int, set region of interest by indices
        region_of_interest_mask[self.target.index.isin(timestamps)] = True

        return region_of_interest_mask

    def get_target_weights(self, roi_weights):
        """
        Set the weights for the region of interest.
        Parameters:
        -----------
        roi_weights: float, np.ndarray
            If float, the weight for each value in the region of interest.
            If np.ndarray of length target, the weights for each
            element in the target
            if np.ndarray of length region of interest, the weights for each element in the region of interest
        """
        if roi_weights is None:
            return np.ones(len(self.target)) / len(self.target)

        elif isinstance(roi_weights, float):
            if not 0 <= roi_weights <= 1:
                raise ValueError("Float weight must be between 0 and 1.")
            # The region of interest shares the weight equally.
            target_weights = np.zeros(len(self.target))
            target_weights[self.region_of_interest] = roi_weights / len(
                target_weights[self.region_of_interest]
            )

            target_weights[~self.region_of_interest] = (1 - roi_weights) / len(
                target_weights[~self.region_of_interest]
            )
            target_weights = target_weights / target_weights.sum()

            return target_weights

        elif isinstance(roi_weights, np.ndarray):
            if len(roi_weights) == len(self.target):
                return roi_weights / roi_weights.sum()
            elif len(roi_weights) == len(self.target[self.region_of_interest]):
                target_weights = np.zeros(len(self.target)) / len(self.target)
                roi_weights_total = roi_weights.sum()
                if not 0 <= roi_weights_total <= 1:
                    raise ValueError(
                        "Weights on region of interest can sum to at most 1."
                    )
                target_weights[self.region_of_interest] = roi_weights
                # the rest of the target equally shares the remaining weight
                target_weights[~self.region_of_interest] = (
                    1 - roi_weights_total
                ) / len(self.target[~self.region_of_interest])
            else:
                raise ValueError(
                    "Weights must be of length target or region of interest."
                )
            return target_weights

    def evaluate(self, forecast, target_weights):
        """
        Evaluate the metric on the target.
        """
        if self.region_of_interest is None:
            raise ValueError("Region of interest is not set.")

        if len(forecast.shape) == 3:
            forecast = forecast[:, :, 0]

        assert self.target.shape[0] == forecast.shape[1]

        base_metric = self.metric(self.target, forecast)

        return np.sum(base_metric * self.target_weights)
