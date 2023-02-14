import torch.cuda
import torch
from dataset import PMDataset
from model import PMLSTM
from torch.utils.data import DataLoader
from preprocess import Preprocessing
from config import VARIABLES_CONSIDERED
import torch.nn as nn
from config import TARGET_VARIABLE, NUMERICAL_VARIABLES, CITIES, TEST_CITY
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class Solution:

    def __init__(self, epochs, batch_size, learning_rate, num_hidden_units, num_layers,
                 dropout, sequence_length):
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.num_hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.__device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.preprocessing_stage = Preprocessing("data")
        self.model = PMLSTM(num_sensors=len(VARIABLES_CONSIDERED) - 3, hidden_units=self.num_hidden_units,
                            num_layers=self.num_layers, dropout=self.dropout)
        if torch.cuda.is_available():
            self.model.to(self.__device)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def create_dataloaders(self, eval: bool = False):
        df_train, df_validation, df_test = self.preprocessing_stage.preprocessing_step()
        cities = CITIES
        train_loaders = []
        validation_loaders = []

        for city in cities:
            if city == TEST_CITY:
                continue
            else:
                train_dataset = PMDataset(
                    df_train.loc[df_train["city"] == city, :].drop("city", axis=1),
                    target=TARGET_VARIABLE,
                    features=NUMERICAL_VARIABLES,
                    sequence_length=self.sequence_length
                )
                validation_dataset = PMDataset(
                    df_validation.loc[df_validation["city"] == city, :].drop("city", axis=1),
                    target=TARGET_VARIABLE,
                    features=NUMERICAL_VARIABLES,
                    sequence_length=self.sequence_length
                )
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False if eval else True)
                validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
                train_loaders.append(train_loader)
                validation_loaders.append(validation_loader)

        test_dataset = PMDataset(
            df_test,
            target=TARGET_VARIABLE,
            features=NUMERICAL_VARIABLES,
            sequence_length=self.sequence_length
        )

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loaders, validation_loaders, test_loader

    def train_model(self):
        train_loaders, _, _ = self.create_dataloaders()
        total_loss = 0
        total_num_batches = 0
        self.model.train()

        for train_loader in train_loaders:
            num_batches = len(train_loader)
            total_num_batches += num_batches

            for X, y in train_loader:
                output = self.model(X)
                loss = self.loss_function(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / total_num_batches

        return avg_loss

    def validate_model(self):
        _, validation_loaders, _ = self.create_dataloaders()
        total_loss = 0
        total_num_batches = 0
        self.model.eval()

        for validation_loader in validation_loaders:

            num_batches = len(validation_loader)
            total_num_batches += num_batches

            with torch.no_grad():
                for X, y in validation_loader:
                    output = self.model(X)
                    total_loss += self.loss_function(output, y).item()

        avg_loss = total_loss / total_num_batches

        return avg_loss

    def train_validation_cycle(self, verbose=False):
        for epoch in range(self.epochs):

            train_loss = self.train_model()
            validation_loss = self.validate_model()
            if verbose:
                print(f"Epoch {epoch}\n---------\n Train loss: {train_loss}, validation loss: {validation_loss}")

    def evaluate_prediction(self):

        train_output = torch.tensor([]).to(self.__device)
        validation_output = torch.tensor([]).to(self.__device)
        self.model.eval()
        train_loaders, validation_loaders, _ = self.create_dataloaders()

        with torch.no_grad():
            for train_loader in train_loaders:
                for X, _ in train_loader:
                    y_pred = self.model(X)
                    train_output = torch.cat((train_output, y_pred), 0)

            for validation_loader in validation_loaders:
                for X, _ in validation_loader:
                    y_pred = self.model(X)
                    validation_output = torch.cat((validation_output, y_pred), 0)

        return train_output.cpu().numpy(), validation_output.cpu().numpy()

    def test_prediction(self):
        test_output = torch.tensor([]).to(self.__device)
        self.model.eval()
        _, _, test_loader = self.create_dataloaders()
        with torch.no_grad():
            for X, _ in test_loader:
                y_pred = self.model(X)
                test_output = torch.cat((test_output, y_pred), 0)
        return test_output.cpu().numpy()

    def plot_predictions_and_present_metrics(self):
        train_output, validation_output = self.evaluate_prediction()
        test_output = self.test_prediction()

        df_train, df_validation, df_test = self.preprocessing_stage.preprocessing_step()
        df_train["y_pred"] = train_output
        df_validation["y_pred"] = validation_output
        df_test["y_pred"] = test_output
        cities = [city for city in CITIES if city != TEST_CITY]

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 10))
        for i, city in enumerate(cities):
            df = df_train.loc[df_train["city"] == city].reset_index()
            axes[i].plot(df.index, df["PM"], label=city)
            axes[i].plot(df.index, df["y_pred"], label="prediction")
            axes[i].xaxis_date()
            axes[i].legend()

        plt.savefig("img/train_predictions.jpeg")
        plt.close()

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 10))
        for i, city in enumerate(cities):
            df = df_validation.loc[df_train["city"] == city].reset_index()
            axes[i].plot(df.index, df["PM"], label=city)
            axes[i].plot(df.index, df["y_pred"], label="prediction")
            axes[i].xaxis_date()
            axes[i].legend()
        plt.savefig("img/validation_predictions.jpeg")
        plt.close()

        df = df_test
        plt.plot(df.index, df["PM"], label=TEST_CITY)
        plt.plot(df.index, df["y_pred"], label="prediction")
        plt.legend()
        plt.savefig("img/test_predictions.jpeg")
        plt.close()

        target_mean = self.preprocessing_stage.target_mean
        target_std = self.preprocessing_stage.target_stdev
        print(f"MSE on the training dataset: {mean_squared_error(df_train['PM']*target_std + target_mean, df_train['y_pred']*target_std + target_mean)}\n")
        print(
            f"MSE on the validation dataset: {mean_squared_error(df_validation['PM'] * target_std + target_mean, df_validation['y_pred'] * target_std + target_mean)}\n")
        print(
            f"MSE on the test dataset: {mean_squared_error(df_test['PM'] * target_std + target_mean, df_test['y_pred'] * target_std + target_mean)}\n")

        return None




