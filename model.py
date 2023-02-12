import torch
import torch.nn as nn


class PMLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers, dropout):
        super().__init__()
        self.num_sensors = num_sensors  # number of features
        self.hidden_units = hidden_units  # number of hidden units
        self.num_layers = num_layers  # number of layers
        self.dropout = dropout  # dropout regularization

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(in_features=self.hidden_units,
                                out_features=1)  # fully-connected layer; it outputs 1 number

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        if torch.cuda.is_available():
            h0 = h0.to(torch.device("cuda:0"))
            c0 = c0.to(torch.device("cuda:0"))
        _, (hn, _) = self.lstm(x, (h0, c0))
        if torch.cuda.is_available():
            hn = hn.to(torch.device("cuda:0"))

        out = self.linear(hn[0]).flatten()  # First dim of hidden layer is num_layers.
        return out
