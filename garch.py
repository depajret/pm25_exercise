from arch import arch_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib
from arch.unitroot import ADF
import statsmodels.graphics.tsaplots as sgt
from datetime import datetime
from datetime import timedelta
import numpy as np


class GARCHSolution:

    def __init__(self, p: int = 1, q: int = 1, city: str = "Beijing", pvalue: float = 0.05,
                 last_obs: str = '2014-12-31', force_diffs: bool = False):
        self.p = p
        self.q = q
        self.pvalue = pvalue
        self.model = None
        self.results = None
        self.city = city
        self.stationary_variable = None
        self.last_obs = datetime.strptime(last_obs, "%Y-%m-%d")
        self.force_diffs = force_diffs


    def prepare_data(self):
        if f"{self.city}.csv" not in os.listdir("data/"):
            return 0
        df = pd.read_csv(pathlib.Path("data", f"{self.city}.csv"))
        pm_cols = [col for col in df.columns if col.startswith("PM")]
        df.loc[:, "PM"] = df[pm_cols].mean(axis=1)
        datetime_cols = ["year", "month", "day", "hour"]
        df["date"] = pd.to_datetime(df[datetime_cols])
        df.set_index("date", inplace=True)
        df.loc[:, "diff"] = df["PM"].pct_change(1)
        df.loc[:, "sq_diff"] = df["diff"] ** 2

        return df[["PM", "diff", "sq_diff"]].dropna()

    def check_augmented_dickey_fuller(self):
        df = self.prepare_data()
        adf = ADF(df["PM"])
        if adf.pvalue < self.pvalue:
            self.stationary_variable = "PM"
        else:
            adf2 = ADF(df["diff"])
            if adf2.pvalue < self.pvalue:
                self.stationary_variable = "diff"

        if self.stationary_variable:
            print(f"Stationary variable based on the ADF test : {self.stationary_variable}")
        else:
            print("You need another way of preprocessing data.")
        return None

    def plot_variable_and_pacf_functions(self):
        df = self.prepare_data()
        self.check_augmented_dickey_fuller()
        df["volatility"] = df[self.stationary_variable] ** 2 if self.stationary_variable else df["sq_diff"]
        if self.stationary_variable:
            df[self.stationary_variable].plot(figsize=(20, 5))
            plt.title(f"{self.stationary_variable}", size=24)
            plt.savefig(f"img/{self.stationary_variable}_plot.jpeg")
            plt.close()
        sgt.plot_pacf(df[self.stationary_variable][1:], lags=40, zero=False, method=('ols'))
        plt.title(f'PACF of {self.stationary_variable}', size=22)
        plt.savefig(f"img/pacf_{self.stationary_variable}.jpeg")
        plt.close()

        df["volatility"].plot(figsize=(20, 5))
        plt.title("Volatility", size=24)
        plt.savefig("img/volatility_plot.jpeg")
        plt.close()

        sgt.plot_pacf(df["volatility"][1:], lags=40, zero=False, method=('ols'))
        plt.title("PACF Volatility", size=22)
        plt.savefig("img/pacf_volatility.jpeg")
        plt.close()
        return None

    def fit_garch_model(self):
        df = self.prepare_data()
        self.check_augmented_dickey_fuller()
        if self.force_diffs:
            model = arch_model(df["diff"].dropna(), p=self.p, q=self.q, vol="GARCH")
        else:
            model = arch_model(df[self.stationary_variable].dropna(), p=self.p, q=self.q, vol="GARCH")
        res = model.fit(last_obs=self.last_obs)
        return model,res

    def forecast_and_look_for_anomalies(self):
        model, res = self.fit_garch_model()

        forecasts = res.forecast(start=(self.last_obs + timedelta(1)), reindex=False)
        cond_mean = forecasts.mean[str((self.last_obs + timedelta(1)).year):]
        cond_var = forecasts.variance[str((self.last_obs + timedelta(1)).year):]
        q = model.distribution.ppf([0.01, 0.05])

        volatility_estimates = -cond_mean.values - np.sqrt(cond_var).values * q[None, :]
        volatility_estimates = pd.DataFrame(volatility_estimates, columns=["1%", "5%"], index=cond_var.index)

        df_forecast = self.prepare_data().loc[str((self.last_obs + timedelta(1)).year):].copy()
        df_forecast["real"] = df_forecast[self.stationary_variable if not self.force_diffs else "diff"]
        df_forecast["confidence_95%"] = volatility_estimates["5%"]
        df_forecast["confidence_99%"] = volatility_estimates["1%"]

        df_forecast.loc[:, "magnitude"] = 0

        df_forecast.loc[df_forecast["real"] >= df_forecast['confidence_99%'], "magnitude"] = 2
        df_forecast.loc[(df_forecast["real"] < df_forecast['confidence_99%']) &
                        (df_forecast["real"] >= df_forecast["confidence_95%"]), "magnitude"] = 1
        len_mag_0 = len(df_forecast.loc[df_forecast['magnitude'] == 0])
        len_mag_1 = len(df_forecast.loc[df_forecast['magnitude'] == 1])
        len_mag_2 = len(df_forecast.loc[df_forecast['magnitude'] == 2])
        print(f"Using model GARCH({self.p},{self.q}), and trained it on the data till {self.last_obs}, \n"
              f"we can say that in the dates that followed,\n {len_mag_0} observations were in 95% confidence interval for volatility,")
        if len_mag_1 > 0 and len_mag_2 > 0:
            print(f"{len_mag_1 + len_mag_0} observations were in 99% confidence interval,\n"
                  f"and only {len_mag_2} exceeded the latter interval.")
        elif len_mag_1 > 0 :
            print(f"and {len_mag_1 + len_mag_0} observations were in 99% confidence interval.")
