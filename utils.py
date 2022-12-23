import torch
import numpy as np


def compute_adjacency_matrix(
    route_distances: np.ndarray, sigma2: float, epsilon: float
):
    num_routes: int = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w_sqaured = route_distances**2
    w_mask = np.ones([num_routes, num_routes]) - np.identity(num_routes)
    return (np.exp(-w_sqaured / sigma2) >= epsilon) * w_mask


class GraphTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        timeseries_data: np.ndarray,
        seq_len: int,
        forecast_horizon: int,
        multi_horizon: bool = False,
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        mu = timeseries_data.mean(axis=0)
        std = timeseries_data.std(axis=0)
        timeseries_data = (timeseries_data - mu) / std

        X = [
            timeseries_data[i : i + seq_len]
            for i in range(0, len(timeseries_data) - seq_len - forecast_horizon, 1)
        ]

        if multi_horizon:
            y = [
                timeseries_data[i : i + forecast_horizon]
                for i in range(seq_len, len(timeseries_data) - forecast_horizon, 1)
            ]

        else:
            y = [
                timeseries_data[i : i + 1]
                for i in range(seq_len + forecast_horizon, len(timeseries_data), 1)
            ]

        self.X = np.stack(X)
        self.y = np.stack(y)
        assert (
            self.X.shape[0] == self.y.shape[0] and self.X.shape[-1] == self.y.shape[-1]
        )

    def __getitem__(self, key):
        return (
            torch.tensor(
                self.X[key], dtype=torch.float32, device=self.device
            ).unsqueeze(-1),
            torch.tensor(
                self.y[key], dtype=torch.float32, device=self.device
            ).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.y)
