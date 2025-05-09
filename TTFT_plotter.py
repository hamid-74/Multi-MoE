import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define paths and initialize data structures
base_folder = "results/qos/"  # Replace with the actual path
approaches = ["multi-MoE", "single", "MIG1", "MIG2"]
runs = [f"{approach}_run{i}" for approach in approaches for i in range(1, 6)]

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 10,
})

# Initialize data storage
metrics_data = {approach: {"TTFT": [], "turnaround_time": [], "weights": []} for approach in approaches}

# Collect turnaround_time and TTFT data
for run in runs:
    if "multi-MoE" in run:
        approach = "multi-MoE"
    elif "single" in run:
        approach = "single"
    elif "MIG1" in run:
        approach = "MIG1"
    elif "MIG2" in run:
        approach = "MIG2"
    else:
        continue

    folder_path = os.path.join(base_folder, run)
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                metrics = json.load(f)
                if metrics["throughput"] > 0:  # Skip files where no requests were handled
                    handled_requests = len(metrics["turnaround_time"])
                    metrics_data[approach]["turnaround_time"].append(metrics["average_turnaround_time"])
                    metrics_data[approach]["TTFT"].append(metrics["average_TTFT"])
                    metrics_data[approach]["weights"].append(handled_requests)

# Compute weighted averages for MIG (combining MIG1 and MIG2)
metrics_data["MIG"] = {"TTFT": [], "turnaround_time": [], "weights": []}
for mig_approach in ["MIG1", "MIG2"]:
    metrics_data["MIG"]["TTFT"].extend(metrics_data[mig_approach]["TTFT"])
    metrics_data["MIG"]["turnaround_time"].extend(metrics_data[mig_approach]["turnaround_time"])
    metrics_data["MIG"]["weights"].extend(metrics_data[mig_approach]["weights"])
del metrics_data["MIG1"]
del metrics_data["MIG2"]

# Calculate weighted averages for each approach
average_metrics = {}
for approach, values in metrics_data.items():
    total_weights = sum(values["weights"])
    if total_weights > 0:
        avg_ttft = np.average(values["TTFT"], weights=values["weights"])
        avg_turnaround_time = np.average(values["turnaround_time"], weights=values["weights"])
        average_metrics[approach] = {"TTFT": avg_ttft, "turnaround_time": avg_turnaround_time}

# Normalize the metrics so the maximum value is 1
max_ttft = max(avg["TTFT"] for avg in average_metrics.values())
max_turnaround = max(avg["turnaround_time"] for avg in average_metrics.values())
for approach in average_metrics:
    average_metrics[approach]["TTFT"] /= max_ttft
    average_metrics[approach]["turnaround_time"] /= max_turnaround

# Plotting
fig, ax = plt.subplots(figsize=(8, 3.5))
width = 0.05 # Bar width
x_indices = np.arange(2)  # Two groups: TTFT and Turnaround Time

x_indices = np.array([0.5, 0.7])
print(x_indices)

# Data for plotting
multi_moe_values = [
    average_metrics["multi-MoE"]["TTFT"],
    average_metrics["multi-MoE"]["turnaround_time"],
]
single_values = [
    average_metrics["single"]["TTFT"],
    average_metrics["single"]["turnaround_time"],
]
mig_values = [
    average_metrics["MIG"]["TTFT"],
    average_metrics["MIG"]["turnaround_time"],
]
print(multi_moe_values)
print(single_values)
print(mig_values)
# Plot bars
ax.bar(x_indices - width, multi_moe_values, width, label="Proposed",  zorder = 3)
ax.bar(x_indices, single_values, width, label="Single Model",  zorder = 3)
ax.bar(x_indices + width, mig_values, width, label="NVIDIA MIG",  zorder = 3)

# Add labels, legend, and grid
# ax.set_xlabel("Metrics")
ax.set_ylabel("Normalized Values")
ax.set_title("TTFT and Turnaround Time Comparison")
ax.set_xticks(x_indices)
ax.set_xticklabels(["TTFT (Max: {:.2f}s)".format(max_ttft), "Turnaround (Max: {:.2f}s)".format(max_turnaround)])
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7, zorder = 0)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('ttft_turnaround.png', dpi=300)
