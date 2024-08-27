""" Data pipeline"""


from typing import Union
import math

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    LabelEncoder,
)

from utils import get_root_dir


def get_scaler(scaler: str):
    scalers = {
        "MinMax": MinMaxScaler(),
        "MaxAbs": MaxAbsScaler(),
        "Standard": StandardScaler(),
        "Robust": RobustScaler(),
        "Quantile": QuantileTransformer(),
    }
    if scaler not in scalers:
        raise ValueError
    return scalers[scaler]


def write_units(kind: str, config: dict):
    """
    This function takes pre-generated text files with units. Returns a list of strings.
    These are currently generated from the TRAIN/TEST pickle files by generate_units_dk.py
    (also filtered by null values and window length) and are specified in config_data_dk.yaml.
    """   
    units_train = get_root_dir().joinpath(config['units_files']['units_for_training'])
    units_test = get_root_dir().joinpath(config['units_files']['units_for_testing'])
    if kind == 'train':
        units = open(units_train).read().splitlines()
    elif kind == 'test':
        units = open(units_test).read().splitlines()

    return units[:100] #PO: using the first 100 units for now


class DatasetImporterDK(object):
    """
    This uses train and test sets as given.
    To compare with the results from ["Unsupervised scalable representation learning for multivariate time series"]
    """
    def __init__(self, units:list, kind:str, data_scaling: bool, config: dict):
        """
        :param dataset_name: e.g., "ElectricDevices"
        :param data_scaling
        """
        self.data_root = get_root_dir().joinpath(config['data_dir']['data_root'])
        self.dir_scalers = get_root_dir().joinpath(config['data_dir']['data_root'], "scalers")

        static_real = config['static_real']
        static_cat = config['static_cat']
        dynamic_cat = config['dynamic_cat']
        dynamic_real = config['dynamic_real']
        col_for_groupby = config['col_for_groupby']
        time_col = config['time_col']
        #units = config['units']

        # fetch an entire dataset
        if kind == 'train':
            df = pd.read_pickle(self.data_root.joinpath(config['data_dir']['train_pkl']))
        elif kind == 'test':
            df = pd.read_pickle(self.data_root.joinpath(config['data_dir']['test_pkl']))

        # Filling nan values, TODO: improve this
        df = df[df[col_for_groupby].isin(units)]
        df = df.fillna(0)
        print(kind, df.shape)

        if data_scaling:
            scaler = {}
            if len(static_real) > 0:
                if kind == "train":
                    _scaler = get_scaler(config["scaler"]).fit(df[static_real])
                    scaler["static_real"] = _scaler
                else:
                    pkl_file = open(self.dir_scalers.joinpath("scaler_static_real.pkl"), "rb")
                    scaler["static_real"] = pickle.load(pkl_file)
                    pkl_file.close()
                df[static_real] = scaler["static_real"].transform(df[static_real])

            for cat in static_cat:
                if kind == "train":
                    labelencoder = LabelEncoder().fit(df[cat])
                    scaler["static_cat"] = labelencoder
                else:
                    pkl_file = open(self.dir_scalers.joinpath("scaler_static_cat.pkl"), "rb")
                    scaler["static_cat"] = pickle.load(pkl_file)
                    pkl_file.close()
                df[cat] = scaler["static_cat"].transform(df[cat])

            for cat in dynamic_cat:
                if df[cat].dtype != int:
                    if kind == "train":
                        labelencoder = LabelEncoder().fit(df[cat])
                        scaler["dynamic_cat"] = labelencoder
                    else:
                        pkl_file = open(self.dir_scalers.joinpath("scaler_dynamic_cat.pkl"), "rb")
                        scaler["dynamic_cat"] = pickle.load(pkl_file)
                        pkl_file.close()
                    df[cat] = scaler["dynamic_cat"].transform(df[cat])

            # Create a list of dataframes, one for each unit (i.e. cell or sector or sensor etc.)
            df = df.reset_index().sort_values(by=[col_for_groupby, time_col])
            df["group_lengths"] = df.groupby(col_for_groupby, sort=False)[
                col_for_groupby
            ].transform("count")
            group_lengths = df["group_lengths"].unique()

            df["group_index"] = df.groupby(col_for_groupby, sort=False).cumcount()
            n_dyn_real_cols = len(dynamic_real)
            if n_dyn_real_cols > 0:
                dfs = []
                for length in group_lengths:
                    # for split in ["train", "val", "test"]:
                    df_part = df.loc[df["group_lengths"] == length].copy()
                    n_units = df_part[col_for_groupby].nunique()
                    dynamic_real_arr = (
                        df_part[dynamic_real]
                        .values.reshape(n_units, -1, n_dyn_real_cols)
                        .swapaxes(0, 1)
                        .reshape(-1, n_dyn_real_cols * n_units)
                    )
                    if kind == "train":
                        scaler["dynamic_real"] = get_scaler(config["scaler"]).fit(
                            dynamic_real_arr
                        )
                    else:
                        pkl_file = open(self.dir_scalers.joinpath("scaler_dynamic_real.pkl"), "rb")
                        scaler["dynamic_real"] = pickle.load(pkl_file)
                        pkl_file.close()
                    dynamic_real_arr = scaler["dynamic_real"].transform(dynamic_real_arr)
                    df_part[dynamic_real] = (
                        dynamic_real_arr.swapaxes(0, 1)
                        .reshape(n_units, n_dyn_real_cols, -1)
                        .swapaxes(1, 2)
                        .reshape(-1, n_dyn_real_cols)
                    )
                    dfs.append(df_part)
            dataframe = pd.concat(dfs)
            df = dataframe.drop(columns=["group_index", "group_lengths"]).set_index(
                time_col
            )

            if kind == "train":
                for split in scaler:
                    output = open(self.dir_scalers.joinpath(f"scaler_{split}.pkl"), "wb")
                    pickle.dump(scaler[split], output)
                    output.close()

            self.df = df
            self.X = self.df[config['dynamic_real']].values.astype(np.float32)
            self.Y = self.df[config['col_for_groupby']].values

class DKDataset(Dataset):
    def __init__(self,
                 kind: str,
                 dataset_importer: DatasetImporterDK,
                 **config,
                 ):
        """
        :param kind: "train" / "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        self.kind = kind
        self.window_len = config['window_len']
        self.window_stride = config['window_stride']
        col_for_groupby = config['col_for_groupby']

        self.df = dataset_importer.df
        self.X = dataset_importer.X
        self.Y = dataset_importer.Y

        if kind == 'train':
            print('self.X_train.shape:', self.X.shape)
            print("# unique labels (train):", np.unique(self.Y.reshape(-1)))
        elif kind == 'test':
            print('self.X_test.shape:', self.X.shape)
            print("# unique labels (test):", np.unique(self.Y.reshape(-1)))


        group_index = self.df.groupby(col_for_groupby, sort=False).cumcount().values
        adj_len = self.adjust_length(
            self.df.groupby(col_for_groupby, sort=False)[col_for_groupby]
            .transform("count")
            .values
        )
        valid_index = (group_index <= adj_len) & ((group_index % self.window_stride) == 0)
        temp = self.df.loc[:, col_for_groupby:col_for_groupby].copy()
        temp["valid_index"] = valid_index
        self.indexes_per_group = (
            temp.groupby(col_for_groupby, sort=False)
            .agg({"valid_index": sum})
            .iloc[:, 0]
            .cumsum()
            .values
        )
        self.index = np.arange(len(group_index))[valid_index]

    def adjust_length(self, length):
        adj_length = length - self.window_len
        return adj_length - np.abs(adj_length % self.window_stride)

    def __getitem__(self, index):
        index = self.index[index]
        end = index + self.window_len
        x = self.X[index:end].reshape(self.window_len)
        y = np.array([self.Y[index]])
        x = x[None, :]  # adds a channel dim
        #print("y shape", y.shape)
        return x, y

    def __len__(self):
        return len(self.index)
    
    



