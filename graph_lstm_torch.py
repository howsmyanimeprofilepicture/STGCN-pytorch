import random
import pandas as pd
import constants
import torch
import numpy as np
from utils import GraphTimeSeriesDataset, compute_adjacency_matrix
from models import GCN_LSTM
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from collections import namedtuple
from tqdm import tqdm


def main():
    __dirname = Path(__file__).parent
    with open(__dirname / "config.yaml") as f:
        config = yaml.safe_load(f)
        Argument = namedtuple("Argument", list(config.keys()))
        args = Argument(*config.values())

    torch.random.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    random.seed(args.SEED)

    route_distances = pd.read_csv("PeMS-M/W_228.csv", header=None).to_numpy()
    route_distances = route_distances[constants.SAMPLE_ROUTES][
        :, constants.SAMPLE_ROUTES
    ]
    speeds_array = pd.read_csv("PeMS-M/V_228.csv", header=None).to_numpy()
    speeds_array = speeds_array[:, constants.SAMPLE_ROUTES]

    split_idx = int(len(speeds_array) * args.TRAIN_VALID_RATIO)
    train_speeds_array = speeds_array[:split_idx]
    test_speeds_array = speeds_array[split_idx:]

    train_data = GraphTimeSeriesDataset(
        timeseries_data=train_speeds_array,
        seq_len=args.SEQ_LEN,
        forecast_horizon=args.FORECAST_HORIZON,
        multi_horizon=args.MULTI_HORIZON,
    )

    test_data = GraphTimeSeriesDataset(
        timeseries_data=test_speeds_array,
        seq_len=args.SEQ_LEN,
        forecast_horizon=args.FORECAST_HORIZON,
        multi_horizon=args.MULTI_HORIZON,
    )

    adj_mat = compute_adjacency_matrix(route_distances, sigma2=0.1, epsilon=0.5)
    model = GCN_LSTM(
        input_dim=args.INPUT_DIM,
        conv_output_dim=args.CONV_OUTPUT_DIM,
        lstm_hid_dim=args.LSTM_HID_DIM,
        adj_mat=adj_mat,
        forecast_horizon=args.FORECAST_HORIZON,
        multi_horizon=args.MULTI_HORIZON,
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=0.0005)
    losses = []
    for X, y in tqdm(train_loader, total=len(train_loader)):
        pred_y = model(X)
        loss = F.mse_loss(pred_y, y)
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    fig, ax = plt.subplots()
    ax.plot(losses)
    fig.savefig("trainloss.png")

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

    model.eval()
    y_preds = []
    y_trues = []
    losses = []
    with torch.no_grad():
        for X, y in test_loader:
            pred_y = model(X)
            loss = F.mse_loss(pred_y, y)
            losses.append(loss.item())
            y_preds.append(pred_y.cpu().numpy()[0, :, 0, :].reshape(-1))
            y_trues.append(y.cpu().numpy()[0, :, 0, :].reshape(-1))
    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)

    fig, ax = plt.subplots()
    ax.plot(y_preds[:5000], alpha=0.5)
    ax.plot(y_trues[:5000], alpha=0.5)
    fig.savefig("prediction.png")


if __name__ == "__main__":
    main()
