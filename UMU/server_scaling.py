import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load server configuration
with open("server_config.json", "r") as f:
    server_config = json.load(f)

THRESHOLD = server_config["THRESHOLD"]  # Task threshold per server
MAX_SERVERS = server_config["MAX_SERVERS"]

# Ensure results directory exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Load traffic tracefile
def load_tracefile(filepath):
    df = pd.read_csv(filepath)
    return df["flow"].values   

# Function to manage servers based on actual traffic
def manage_servers(actual_traffic):
    active_servers = 1  # Start with one server
    server_activations = []
    for load in actual_traffic:
        if load > THRESHOLD * active_servers and active_servers < MAX_SERVERS:
            active_servers += 1
        elif load < THRESHOLD * (active_servers - 1) and active_servers > 1:
            active_servers -= 1
        server_activations.append(active_servers)
    print(f"Final Active servers: {active_servers}")
    return server_activations

# Example usage
tracefile_path = "../FDI_LSTM/filtered_traffic_data.csv"
data = load_tracefile(tracefile_path)
server_activations = manage_servers(data)

# Plot server activations over time
plt.figure(figsize=(10, 5))
plt.plot(server_activations, label="Active Servers", color='blueviolet')
plt.xlabel("Steps")
plt.ylabel("Number of Active Servers")
plt.title("Server Activations Over Time")
plt.legend()
plt.grid(True)  
plt.savefig(os.path.join(results_dir, "server_activations.pdf"), format="pdf")  
plt.show()
