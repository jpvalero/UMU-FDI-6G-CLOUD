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

adaptive_thresholds = [] 

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
        adaptive_thresholds.append(adaptive_threshold) 

        server_activations.append(active_servers)
    return server_activations

# Function to plot server activations
def plot_server_activations(actual_activations, predicted_activations_adaptive, predicted_activations_fixed):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_activations, label="[Reactive, Without ML]", color="#AEC6CF")  # Pastel blue
    plt.plot(predicted_activations_fixed, label="[Proactive, Fixed Threshold]", color="#77DD77", linestyle="--")  # Pastel green
    plt.plot(predicted_activations_adaptive, label="[Proactive, Adaptive Threshold]", color="#FFB347", linestyle="--")  # Pastel orange
    plt.xlabel("Steps")
    plt.ylabel("Number of Active Servers")
    plt.ylim(0, 25)
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(os.path.join(results_dir, "server_activation_comparison.pdf"), format="pdf")
    plt.show()

# Process actual traffic data with fixed threshold
actual_tracefile_path = "data/filtered_traffic_data.csv"
actual_data = load_tracefile(actual_tracefile_path)
actual_data = actual_data[994:len(actual_data)]  # 994 to the end to take half the data
actual_server_activations = manage_servers_fixed(actual_data, THRESHOLD)

# Process predicted traffic data with adaptive and fixed thresholds
predicted_tracefile_path = "data/predictions.csv"
predicted_data = load_tracefile(predicted_tracefile_path)
predicted_server_activations_adaptive = manage_servers_adaptive(predicted_data)
predicted_server_activations_fixed = manage_servers_fixed(predicted_data, THRESHOLD)

# Compute power consumption
actual_power_consumption = [s * POWER_SERVER for s in actual_server_activations]
predicted_power_consumption_adaptive = [s * POWER_SERVER for s in predicted_server_activations_adaptive]
predicted_power_consumption_fixed = [s * POWER_SERVER for s in predicted_server_activations_fixed]

# Print mean number of active servers and power consumption
actual_mean = sum(actual_server_activations) / len(actual_server_activations)
predicted_mean_adaptive = sum(predicted_server_activations_adaptive) / len(predicted_server_activations_adaptive)
predicted_mean_fixed = sum(predicted_server_activations_fixed) / len(predicted_server_activations_fixed)
actual_power_mean = sum(actual_power_consumption) / len(actual_power_consumption)
predicted_power_mean_adaptive = sum(predicted_power_consumption_adaptive) / len(predicted_power_consumption_adaptive)
predicted_power_mean_fixed = sum(predicted_power_consumption_fixed) / len(predicted_power_consumption_fixed)

print(f"Mean number of active servers (Actual): {actual_mean:.2f}")
print(f"Mean number of active servers (Predicted, Adaptive Threshold): {predicted_mean_adaptive:.2f}")
print(f"Mean number of active servers (Predicted, Fixed Threshold): {predicted_mean_fixed:.2f}")
print(f"Mean power consumption (Actual): {actual_power_mean:.2f} W")
print(f"Mean power consumption (Predicted, Adaptive Threshold): {predicted_power_mean_adaptive:.2f} W")
print(f"Mean power consumption (Predicted, Fixed Threshold): {predicted_power_mean_fixed:.2f} W")

# Plot server activations
plot_server_activations(actual_server_activations, predicted_server_activations_adaptive, predicted_server_activations_fixed)

# Bar plot for power consumption comparison
plt.figure(figsize=(10, 6)) 
labels = ["[Reactive, Without ML]", "[Proactive, Fixed Threshold]", "[Proactive, Adaptive Threshold]"] 
values = [actual_power_mean, predicted_power_mean_fixed, predicted_power_mean_adaptive, ]
plt.bar(labels, values, color=["#B19CD9", "#FF6961", "#77DD77"])  # Pastel purple, pastel red, pastel green
plt.ylabel("Mean Power Consumption (W)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid() 
plt.savefig(os.path.join(results_dir, "power_consumption_comparison.pdf"), format="pdf")
plt.show()
