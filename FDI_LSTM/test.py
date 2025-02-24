import os
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from utils import *
from sklearn.preprocessing import StandardScaler

# Load hyperparameters from JSON file
with open('hyperparameters.json', 'r') as f:
    hyperparameters = json.load(f)

# Setup testing configurations
BATCH_SIZE = hyperparameters['BATCH_SIZE']
TIME_STEP = hyperparameters['TIME_STEP']
COLUMN = hyperparameters['COLUMN']  # 'cpu_consumption'
ROOT_FOLDER = hyperparameters['ROOT_FOLDER']

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
csv_file = os.path.join(ROOT_FOLDER, 'filtered_traffic_data.csv')
df = pd.read_csv(csv_file)
df_y = preprocess(df[COLUMN])

# Load trained model
model = LSTMModel(input_size=TIME_STEP, hidden_layer_size=hyperparameters['HIDDEN_LAYER_SIZE'], num_layers=hyperparameters['NUM_LAYERS'], output_size=1).to(device)
model.load_state_dict(torch.load('save_folder/model.pt'))
model.eval()

# Load the scaler
scaler = joblib.load('save_folder/scaler.pkl')

# Split dataset
train_split, test_split = train_test_split(df_y, hyperparameters['TRAIN_TEST_SPLIT'])
test_dataset = LoadPredictionDataset(df_y, time_step=TIME_STEP, column=COLUMN, start_index=train_split, population=test_split, device=device)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Plot results
split_index = int(hyperparameters['TRAIN_TEST_SPLIT']*len(df)) + 1
plot_results(model, test_dataset, test_dataloader, time_step=TIME_STEP, device=device, save_folder='save_folder', original_df=df.iloc[split_index:].reset_index(drop=True), scaler=scaler)
