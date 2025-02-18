import os 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json
import joblib  # Import joblib for loading the scaler

from utils import *
bookkeeper = Bookkeeper()
run_folder = bookkeeper.get_run_folder()
hyperparameters = bookkeeper.get_hyperparameters()

BATCH_SIZE = hyperparameters['BATCH_SIZE']
FILL_NAN = hyperparameters['FILL_NAN']
TIME_STEP = hyperparameters['TIME_STEP']
COLUMN = hyperparameters['COLUMN']  # 'cpu_consumption'
EPOCHS = hyperparameters['EPOCHS']
lr = hyperparameters['lr'] 
HIDDEN_LAYER_SIZE = hyperparameters['HIDDEN_LAYER_SIZE']
NUM_LAYERS = hyperparameters['NUM_LAYERS']
TRAIN_TEST_SPLIT = hyperparameters['TRAIN_TEST_SPLIT']
ROOT_FOLDER = hyperparameters['ROOT_FOLDER']

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! Using GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")  # Use GPU
else:
    print("CUDA not available, using CPU.")
    device = torch.device("cpu")  # Use CPU

csv_file = os.path.join(ROOT_FOLDER, 'filtered_traffic_data.csv')
df = pd.read_csv(csv_file)
df = df[COLUMN]
df = preprocess(df)

train_split, test_split = train_test_split(df, TRAIN_TEST_SPLIT)
train_dataset = LoadPredictionDataset(df, time_step=TIME_STEP, column=COLUMN, start_index=0, population=train_split, device=device)
test_dataset = LoadPredictionDataset(df, time_step=TIME_STEP, column=COLUMN, start_index=train_split, population=test_split, device=device)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the pre-trained scaler
scaler_file = 'save_folder/scaler.pkl'  # Load the scaler from the file where it was saved
scaler = joblib.load(scaler_file)

model = LSTMModel(input_size=TIME_STEP, hidden_layer_size=HIDDEN_LAYER_SIZE, num_layers=NUM_LAYERS, output_size=1).to(device)

criterion = nn.MSELoss()  # or any other loss function based on your task
optimizer = optim.Adam(model.parameters(), lr=lr)  # or any other optimizer

train_lstm(model, epochs=EPOCHS, train_dataloader=train_dataloader, criterion=criterion, optimizer=optimizer)

plot_results(model,test_dataset,test_dataloader,time_step=TIME_STEP)