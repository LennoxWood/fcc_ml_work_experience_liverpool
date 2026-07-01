"""
dataset.py

Wraps EventsDataset to add per-feature standardisation (normalisation).
Import this in the notebook rather than defining the class there.

Usage:
    from dataset import CustomEventsDataset
    dataset = CustomEventsDataset(root=..., url=..., ...)
    train_indices, test_indices = train_test_split(...)
    dataset.fit_normalization(train_indices)   # must call BEFORE iterating
    # then access dataset[i].data_norm in the training loop
"""

import numpy as np
import torch
from sparticles.dataset_edge_embed import EventsDataset


class CustomEventsDataset(EventsDataset):
    """
    Extends EventsDataset with per-feature z-score normalisation.

    After loading the dataset and splitting into train/test, call
    fit_normalization(train_indices) once. This computes mean and std
    using only the training events (no data leakage into the test set).
    Each item returned by __getitem__ then has a data.data_norm attribute
    containing the normalised feature matrix, ready for the model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_values = None
        self.std_values  = None

    def fit_normalization(self, indices):
        """
        Compute mean and std from the given indices (training set only).
        Must be called before any item is accessed via __getitem__.
        """
        all_features = np.concatenate(
            [self.get(idx).x.numpy() for idx in indices], axis=0
        )
        mean = np.nanmean(all_features, axis=0)
        std  = np.nanstd(all_features,  axis=0)

        # After MakeHomogeneous the feature vector is doubled into
        # (value, is_nan_flag) interleaved pairs. Insert a 0 after each
        # mean (so mask features aren't shifted) and ~1 after each std
        # (so they aren't rescaled).
        def interleave(arr, fill):
            result = []
            for v in arr:
                result.extend([v, fill])
            return result

        self.mean_values = torch.tensor(interleave(mean, 0),        dtype=torch.float)
        self.std_values  = torch.tensor(interleave(std,  0.999999), dtype=torch.float)

    def _normalise(self, data):
        if self.mean_values is None:
            raise RuntimeError(
                "Call dataset.fit_normalization(train_indices) before "
                "accessing any dataset items."
            )
        return (data.x - self.mean_values) / (self.std_values + 1e-6)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.data_norm = self._normalise(data)
        return data