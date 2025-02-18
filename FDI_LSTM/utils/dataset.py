import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib  # Import joblib for saving the scaler

def preprocess(df):
    df = df.to_numpy()
    df = df.reshape(-1,1)
    scaler = StandardScaler()
    df  = pd.DataFrame(scaler.fit_transform(df))
    
    # Save the scaler to a file so it can be loaded later
    scaler_file = 'save_folder/scaler.pkl'  # You can adjust this path as needed
    joblib.dump(scaler, scaler_file)  # Save the scaler
    
    return df


def train_test_split(df,split):
    train_split = int(len(df)*split)
    test_split = len(df) - train_split
    return train_split, test_split

class LoadPredictionDataset(Dataset):
    def __init__(self, df, start_index, population, time_step, column, device):
        previous_overflow = max(start_index - time_step, 0)
        self.df = df.iloc[previous_overflow:start_index + population]
        self.column = column
        self.time_step = time_step
        self.device = device
        self.length = len(self.df) - self.time_step - 1
        
    def __getitem__(self, index):
        previous_values = self.df.iloc[index:index + self.time_step].values
        previous_values = torch.tensor(previous_values)
        previous_values = previous_values.float().to(self.device)
        previous_values = previous_values.view(1, -1)
        target_values = self.df.iloc[index + self.time_step]
        target_values = torch.tensor(target_values).float().to(self.device)
        target_values = target_values
        return previous_values, target_values
    
    def __len__(self):
        return self.length
