import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def plot_results(model, dataset, dataloader, time_step, device=None, save_folder="save_folder", original_df=None, scaler=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = torch.zeros(time_step, 1).to(device)
    
    mses = []
    loss_fn = torch.nn.MSELoss()
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No need to compute gradients
        for data, ans in dataloader:
            data = data.to(device)
            ans = ans.to(device)
            res = model(data)
            results = torch.cat((results, res), dim=0)
            mses.append(loss_fn(res, ans))
    
    avg_mse = sum(mses) / len(mses)
    print(f"Mean Squared Error: {avg_mse.item()}")

    plt.figure(figsize=(10, 5))
    plt.plot(results.detach().cpu().numpy()[time_step:], color='r', label='Predicted')
    plt.plot(dataset.df.values[time_step:], color='b', label='Real')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("LSTM Predictions vs Real Data")
    plt.tight_layout()
    
    if scaler:
        results_np = scaler.inverse_transform(results.detach().cpu().numpy()[time_step:])
        result_df = pd.DataFrame(data= results_np, columns=['Predicted'])
    else:
        result_df = pd.DataFrame(data= results.detach().cpu().numpy()[time_step:], columns=['Predicted'])

    if save_folder is None:
        plt.show()
    else:
        os.makedirs(save_folder, exist_ok=True)  # Ensure directory exists
        test_file = os.path.join(save_folder, 'test.png')
        plt.savefig(test_file)
        if original_df is not None:
            joined_df = original_df.join(result_df)
            result_csv = os.path.join(save_folder, 'predictions.csv')
            joined_df.to_csv(result_csv, index=False)
        else: 
            result_csv = os.path.join(save_folder, 'predictions.csv')
            result_df.to_csv(result_csv, index=False)
        
    plt.clf()



def train_lstm(model, epochs, train_dataloader, criterion, optimizer, save_folder="save_folder", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    losses = []
    for epoch in range(epochs):
        accumulative_loss = 0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            accumulative_loss += loss.item()
        losses.append(accumulative_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss: {accumulative_loss / len(train_dataloader)}")

    print('Finished Training')
    plt.plot(losses)
    if save_folder is None:
        plt.show()
    else:
        loss_file = save_folder + '/loss.png'
        plt.savefig(loss_file)
        model_file = save_folder + '/model.pt'
        torch.save(model.state_dict(), model_file)  # Save the model state dict
    plt.clf()
