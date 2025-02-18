import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

def plot_results(model, dataset, dataloader, time_step, device=None, save_folder="save_folder"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    results = torch.zeros(time_step, 1).to(device)
    mses = []
    loss = nn.MSELoss()
    
    # Collect all the results
    for data, ans in dataloader:
        data = data.to(device)
        ans = ans.to(device)
        res = model(data)
        results = torch.cat((results, res), dim=0)
        mses.append(loss(res, ans))
    
    # Calculate MSE and print it
    print(sum(mses) / len(mses))
    
    # Remove initial zeros from the results (trim the first `time_step` values)
    trimmed_results = results[time_step:]  # Remove zeros at the start
    
    # Plotting
    plt.plot(trimmed_results.detach().cpu(), color='r', label='Predicted')
    plt.plot(dataset.df.values[time_step:], color='b', label='Real')  # Offset real values as well
    
    plt.legend()
    plt.tight_layout()
    
    # Save or show the plot
    if save_folder is None:
        plt.show()
    else:
        test_file = save_folder + '/test.png'
        plt.savefig(test_file)
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
