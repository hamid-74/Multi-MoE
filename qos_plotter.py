import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Define paths and initialize data structures
base_folder = "results/qos/"  # Replace with the actual path
approaches = ["multi-MoE", "single", "MIG1", "MIG2"]
runs = [f"{approach}_run{i}" for approach in approaches for i in range(1, 6)]
rates = [round(0.005 * i, 3) for i in range(1, 17)]  # 0.005 to 0.080

plt.rcParams['font.family'] = 'serif'
# Data dictionary to store throughput for each approach and rate
data = {approach: {rate: [] for rate in rates} for approach in approaches}

# Collect throughput data
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
    for rate in rates:
        file_name = f"qos_metrics_rate_{rate:.3f}.json"
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                metrics = json.load(f)
                data[approach][rate].append(metrics["throughput"])

# Compute average throughput for each approach and rate
average_throughput = {approach: {rate: np.mean(data[approach][rate]) for rate in rates} for approach in approaches}

# Combine MIG1 and MIG2 into a single "MIG" bar
average_throughput["MIG"] = {rate: (average_throughput["MIG1"][rate] + average_throughput["MIG2"][rate]) 
                             for rate in rates}
del average_throughput["MIG1"]
del average_throughput["MIG2"]

# Plotting
x = np.array(rates)
width = 0.25  # Bar width

fig, ax = plt.subplots(figsize=(10, 4))

# Bar positions
x_indices = np.arange(len(rates))
multi_moe_throughput = [average_throughput["multi-MoE"][rate] for rate in rates]
single_throughput = [average_throughput["single"][rate] for rate in rates]
mig_throughput = [average_throughput["MIG"][rate] for rate in rates]

# Plot bars with zorder > grid's zorder
ax.bar(x_indices - width, multi_moe_throughput, width, label="Proposed", zorder=3)
ax.bar(x_indices, single_throughput, width, label="Single Model", zorder=3)
ax.bar(x_indices + width, mig_throughput, width, label="NVIDIA MIG", zorder=3)

# Add labels, legend, and grid
ax.set_xlabel("Arrival Rate (request/sec)")
ax.set_ylabel("Throughput (handled requests/minute)")
ax.set_title("Throughput Comparison")
ax.set_xticks(x_indices)
ax.set_xticklabels([f"{rate:.3f}" for rate in rates], rotation=45)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)  # Ensure grid lines have lower zorder

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('qos.png', dpi=300)
