import os
import pandas
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.mypath import MyPath
import ast
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class SWAT(Dataset):
    """`SMD <https://www>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ```` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a ts
            and returns a transformed version.
    """
    base_folder = ''

    def __init__(self, fname, root=MyPath.db_root_dir('swat'), train=True, transform=None, sanomaly= None, mean_data=None, std_data=None):

        super(SWAT, self).__init__()
        if 'swat_DATASET_PATH' in os.environ:
            root = os.environ['swat_DATASET_PATH']

        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train  # training set or test set
        self.classes = ['Normal', 'Anomaly']

        self.data = []
        self.targets = []
        labels = []
        wsz, stride = 256, 50

        if self.train:
            file_path = os.path.join(self.root, "normal.csv")
        else:
            file_path = os.path.join(self.root, "attack.csv")

        temp = pd.read_csv(file_path)
        
        # Clean up column names (strip whitespace)
        temp.columns = temp.columns.str.strip()
        
        # Detect label column
        label_col = 'attack'
        possible_labels = ['attack', 'Attack', 'Normal/Attack', 'label', 'class']
        for col in possible_labels:
            if col in temp.columns:
                label_col = col
                break
        
        if label_col not in temp.columns:
            print(f"Warning: Label column not found. Available columns: {temp.columns.tolist()}")
            # Fallback: assume the last column is the label if not found
            label_col = temp.columns[-1]
            print(f"Using last column '{label_col}' as label.")

        # Map labels to 0 (Normal) and 1 (Attack)
        # Assuming 'Normal' is 0 or 'Normal', 'Attack' is 1 or 'Attack'
        temp[label_col] = temp[label_col].apply(lambda x: 1 if str(x).lower() in ['attack', '1', 'anomaly'] else 0)
        labels = np.asarray(temp[label_col])
        
        # Robust Feature Extraction
        # Drop Timestamp and Label columns to ensure only sensors remain
        # User defined format: Timestamp, Sensor1, ..., Sensor51, Label
        drop_cols = ['Timestamp', 'timestamp', 'Date', 'Time', label_col]
        
        # Also drop 'Normal/Attack' if it exists and wasn't picked as label_col
        if 'Normal/Attack' in temp.columns and 'Normal/Attack' != label_col:
            drop_cols.append('Normal/Attack')

        temp = temp.drop(columns=drop_cols, errors='ignore')

        # print(f"Features columns: {temp.columns.tolist()}")

        temp = np.asarray(temp).astype(np.float32)

        if np.any(sum(np.isnan(temp))!=0):
            print('Data contains NaN which replaced with zero')
            temp = np.nan_to_num(temp)

        self.mean, self.std = mean_data, std_data

        if self.train:
            min_column = np.amin(temp, axis=0)
            max_column = np.amax(temp, axis=0)
            self.mean, self.std = min_column, max_column 
            range_val = (max_column - min_column) + 1e-20
            temp = (temp - min_column) / range_val 
        else:
            # For test set, use the statistics from training set
            # If mean_data and std_data are provided (which they should be for test)
            if mean_data is None or std_data is None:
                 print("Warning: Test set loaded without mean_data/std_data. Using self-statistics (may cause leakage/shift).")
                 min_column = np.amin(temp, axis=0)
                 max_column = np.amax(temp, axis=0)
                 self.mean, self.std = min_column, max_column
            else:
                 self.mean, self.std = mean_data, std_data
            
            range_val = (self.std - self.mean) + 1e-20
            temp = (temp - self.mean) / range_val

        self.targets = labels
        self.data = np.asarray(temp)
        self.data, self.targets = self.convert_to_windows(wsz, stride)

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.data.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.data[st:st+w_size]
            if self.targets[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        ts_org = torch.from_numpy(self.data[index]).float() # cuda
        if len(self.targets) > 0:
            target = torch.tensor(self.targets[index].astype(int), dtype=torch.long)
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        ts_size = (ts_org.shape[0], ts_org.shape[1])

        out = {'ts_org': ts_org, 'target': target, 'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}

        return out

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def get_info(self):
        return self.mean, self.std

    def concat_ds(self, new_ds):
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")