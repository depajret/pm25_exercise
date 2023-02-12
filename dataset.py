import torch
from torch.utils.data import Dataset
import pandas


class PMDataset(Dataset):
    def __init__(self, dataframe: pandas.DataFrame, target: list, features: list,
                 sequence_length: int = 5):
        self.features = features  # exogeneous features
        self.target = target  # target variable
        self.sequence_length = sequence_length  # the "lag" for features and target
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        if torch.cuda.is_available():
            self.y = self.y.to(torch.device("cuda:0"))
            self.X = self.X.to(torch.device("cuda:0"))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        if torch.cuda.is_available():
            x = x.to(torch.device("cuda:0"))

        return x, self.y[i]
