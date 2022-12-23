import torch
import torch.nn as nn


class GCN_LSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        conv_output_dim,
        lstm_hid_dim,
        adj_mat,
        forecast_horizon: int,
        multi_horizon: bool,
        last_num_layers=2,
        device=None,
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.gcn = GraphConvolutionLayer(
            input_dim=input_dim,
            output_dim=conv_output_dim,
            adj_mat=adj_mat,
            device=device,
        )

        self.lstm = torch.nn.LSTM(
            input_size=conv_output_dim,
            hidden_size=lstm_hid_dim,
            device=device,
            num_layers=last_num_layers,
            batch_first=True,
        )

        self.fc = torch.nn.Linear(
            lstm_hid_dim, forecast_horizon if multi_horizon else 1, device=device
        )

    def forward(self, input):
        gcn_output = self.gcn(input)
        (
            batch_size,
            seq_len,
            num_nodes,
            output_dims,
        ) = gcn_output.size()
        gcn_output = gcn_output.transpose(1, 2)
        gcn_output = gcn_output.reshape(batch_size * num_nodes, seq_len, output_dims)

        whole_hidden_state, (
            last_hiden_state,
            last_cell_state,
        ) = self.lstm(gcn_output)
        output = last_hiden_state[-1, :, :]
        assert output.size(0) == batch_size * num_nodes
        output = self.fc(output)
        output = output.reshape(batch_size, num_nodes, self.fc.out_features, 1)
        output = output.transpose(
            1, 2
        )  # (batch_size, predicted_time_steps, num_nodes, node_dims)
        return output


class GraphConvolutionLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        adj_mat,
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        assert output_dim % 2 == 0, "output_dim must be an even number."
        super().__init__()
        self.adj = torch.tensor(
            adj_mat,
            dtype=torch.float32,
            device=self.device,
        )
        self.input_dim, self.output_dim = input_dim, output_dim
        self.W = torch.nn.Linear(
            input_dim,
            int(output_dim / 2),
            bias=False,
            device=self.device,
        )

    def forward(self, input):
        """Args:
        input: `(batch_size, seq_len, num_nodes, input_dim)`
        """
        assert input.size(-1) == self.input_dim

        weighted_input = self.W(input)
        aggregated_messages = torch.matmul(self.adj, input)
        aggregated_messages = self.W(aggregated_messages)
        output = torch.cat([weighted_input, aggregated_messages], dim=-1)
        assert output.size(-1) == self.output_dim
        assert input.size(0) == output.size(0)
        assert input.size(1) == output.size(1)

        return output
