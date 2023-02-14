import arch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib


class GARCHSolution:

    def __init__(self, p: int = 1, q: int = 1, city: str = "Beijing"):
        self.p = p
        self.q = q
        self.model = None
        self.results = None
        self.city = city


    def train_model(self):
        pass

    def prepare_data(self):
        if f"{self.city}.csv" not in os.listdir("data/"):
            return 0
        df = pd.read_csv(pathlib.Path("data", f"{self.city}.csv"))
        pm_cols = [col for col in df.columns if col.startswith("PM")]
        df["PM"] = df[pm_cols].mean(axis=1)
        datetime_cols = ["year", "month", "day", "hour"]
        df["date"] = pd.to_datetime(df[datetime_cols])
        df.set_index("date", inplace=True)
        df = df.loc[:, "PM"]
        df.loc[:, "diff"] = df["PM"].pct_change(1)
        df.loc[:, "sq_diff"] = df["diff"] ** 2
        return df
