import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load server configuration
with open("server_config.json", "r") as f:
    server_config = json.load(f)

THRESHOLD = server_config["THRESHOLD"]  # Task threshold per server
MAX_SERVERS = server_config["MAX_SERVERS"]
POWER_SERVER = server_config["POWER_SERVER"]  # Power consumption per server

# Ensure results directory exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Load traffic tracefile
def load_tracefile(filepath):
    df = pd.read_csv(filepath)
    return df["flow"].values

# Function to manage servers based on traffic with fixed threshold
def manage_servers_fixed(actual_traffic, threshold):
    active_servers = 1  # Start with one server
    server_activations = []
    
    for load in actual_traffic:
        if load > threshold * active_servers and active_servers < MAX_SERVERS:
            active_servers += 1
        elif load < threshold * (active_servers - 1) and active_servers > 1:
            active_servers -= 1
        
        server_activations.append(active_servers)
    return server_activations

# Function to manage servers based on traffic with adaptive threshold
def manage_servers_adaptive(actual_traffic):
    active_servers = 1  # Start with one server
    server_activations = []
    adaptive_threshold = actual_traffic.mean() * 0.5  # Initial adaptive threshold
    
    for load in actual_traffic:
        if load > adaptive_threshold * active_servers and active_servers < MAX_SERVERS:
            active_servers += 1
        elif load < adaptive_threshold * (active_servers - 1) and active_servers > 1:
            active_servers -= 1
        
        # Adapt threshold based on recent traffic
        adaptive_threshold = (0.9 * adaptive_threshold) + (0.1 * load)  # Smooth adaptation
        
        server_activations.append(active_servers)
    return server_activations

# Function to plot traffic data
def plot_traffic_data(actual_data, predicted_data):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_data, label="Actual Traffic", color="blue")
    plt.plot(predicted_data, label="Predicted Traffic", color="red", linestyle="--")
    plt.xlabel("Steps")
    plt.ylabel("Traffic Flow")
    # plt.title("Actual vs Predicted Traffic")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, "traffic_comparison.pdf"), format="pdf")
    plt.show()

# Process actual traffic data with fixed threshold
actual_tracefile_path = "data/filtered_traffic_data.csv"
actual_data = load_tracefile(actual_tracefile_path)
actual_server_activations = manage_servers_fixed(actual_data, THRESHOLD)

# Process predicted traffic data with adaptive threshold
predicted_tracefile_path = "data/predictions.csv"
predicted_data = load_tracefile(predicted_tracefile_path)
# predicted_server_activations = manage_servers_adaptive(predicted_data)
predicted_server_activations = manage_servers_fixed(predicted_data, THRESHOLD) 

# Plot the traffic data
plot_traffic_data(actual_data, predicted_data)

# Compute power consumption
actual_power_consumption = [s * POWER_SERVER for s in actual_server_activations]
predicted_power_consumption = [s * POWER_SERVER for s in predicted_server_activations]

# Print mean number of active servers and power consumption
actual_mean = sum(actual_server_activations) / len(actual_server_activations)
predicted_mean = sum(predicted_server_activations) / len(predicted_server_activations)
actual_power_mean = sum(actual_power_consumption) / len(actual_power_consumption)
predicted_power_mean = sum(predicted_power_consumption) / len(predicted_power_consumption)
print(f"Mean number of active servers (Actual): {actual_mean:.2f}")
print(f"Mean number of active servers (Predicted, Adaptive Threshold): {predicted_mean:.2f}")
print(f"Mean power consumption (Actual): {actual_power_mean:.2f} W")
print(f"Mean power consumption (Predicted, Adaptive Threshold): {predicted_power_mean:.2f} W")

# Bar plot for power consumption comparison
plt.figure(figsize=(6, 5))
labels = ["Actual", "Predicted"]
values = [actual_power_mean, predicted_power_mean]
plt.bar(labels, values, color=['blueviolet', 'darkorange'])
# plt.xlabel("Method")
plt.ylabel("Mean Power Consumption (W)")
# plt.title("Mean Power Consumption Comparison")
plt.savefig(os.path.join(results_dir, "power_consumption_comparison.pdf"), format="pdf")  
plt.show()