import copy
import pathlib
import pandas
import pandas as pd
import os
from config import TEST_CITY, VALIDATION_DATE, TARGET_VARIABLE, PATTERN_FOR_TARGET, VARIABLES_CONSIDERED


class Preprocessing:
    def __init__(self, data_path):
        self.data_path = data_path
        self.__target_mean = None
        self.__target_stdev = None

    @property
    def target_mean(self):
        return self.__target_mean

    @property
    def target_stdev(self):
        return self.__target_stdev

    def load_data(self, file):
        df = pd.read_csv(pathlib.Path(self.data_path, file))
        return df

    @staticmethod
    def parse_dates(df: pandas.DataFrame):
        df_output = copy.deepcopy(df)
        df_output.loc[:, "date"] = pd.to_datetime(df_output[["year", "month", "day", "hour"]])
        df_output.drop(["year", "month", "day", "hour", "season"], axis=1, inplace=True)
        return df_output

    @staticmethod
    def __select_pm_columns(df: pandas.DataFrame):
        return [col for col in df.columns if str(col).startswith(PATTERN_FOR_TARGET)]

    @staticmethod
    def calculate_pm(df: pandas.DataFrame):
        pm_columns = Preprocessing.__select_pm_columns(df)
        df_output = copy.deepcopy(df)
        df_output.loc[:, TARGET_VARIABLE] = df_output[pm_columns].mean(axis=1)
        df_output.drop(pm_columns, axis=1, inplace=True)
        return df_output

    @staticmethod
    def filter_columns(df: pandas.DataFrame):
        df_output = copy.deepcopy(df)
        df_output = df_output[VARIABLES_CONSIDERED]
        return df_output

    def load_data_batch(self):
        for i, file in enumerate(os.listdir(self.data_path)):
            df = self.load_data(file)
            df.loc[:, "city"] = file[:-4]
            df = Preprocessing.calculate_pm(df)
            df = Preprocessing.parse_dates(df)
            df = Preprocessing.filter_columns(df)

            if i == 0:
                df_output = copy.deepcopy(df)
            else:
                df_output = pd.concat([df_output, df])

            df_output.dropna(inplace=True)

        return df_output

    def split_data_to_train_test_validation(self):
        df = self.load_data_batch()
        df_train_validation = df.loc[df["city"] != TEST_CITY]
        df_test = df.loc[df["city"] == TEST_CITY]
        df_validation = df_train_validation.loc[df_train_validation["date"] >= VALIDATION_DATE]
        df_train = df_train_validation.loc[df_train_validation["date"] < VALIDATION_DATE]
        return df_train, df_validation, df_test

    def standardize_datasets(self, df_train: pandas.DataFrame, df_validation: pandas.DataFrame,
                             df_test: pandas.DataFrame):
        df_train_output = copy.deepcopy(df_train)
        df_validation_output = copy.deepcopy(df_validation)
        df_test_output = copy.deepcopy(df_test)

        for c in df_train_output.columns:
            mean = df_train[c].mean()
            stdev = df_train[c].std()

            if c == TARGET_VARIABLE:
                self.__target_mean = mean
                self.__target_stdev = stdev

            df_train_output[c] = (df_train_output[c] - mean) / stdev
            df_validation_output[c] = (df_validation_output[c] - mean) / stdev
            df_test_output[c] = (df_test_output[c] - mean) / stdev

        return df_train_output, df_validation_output, df_test_output

    def preprocessing_step(self):
        df_train, df_validation, df_test = self.split_data_to_train_test_validation()
        df_train.drop(["city", "date"], axis=1, inplace=True)
        df_validation.drop(["city", "date"], axis=1, inplace=True)
        df_test.drop(["city", "date"], axis=1, inplace=True)

        df_train, df_validation, df_test = self.standardize_datasets(df_train, df_validation, df_test)

        return df_train, df_validation, df_test
