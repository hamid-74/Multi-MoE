import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set global font to Times New Roman
rcParams['font.family'] = 'serif'

# Load the 2D JSON array from the file
with open('expert_dist.json', 'r') as f:
    data = json.load(f)

# Convert the JSON list to a NumPy array
array = np.array(data)

# Transpose the array to switch axes
array_transposed = array.T

# Plot the heatmap
plt.figure(figsize=(16, 4))
plt.imshow(array_transposed, cmap='viridis', aspect='auto')

# Add a colorbar for reference
plt.colorbar(label='L2 Distance')

# Set x and y axis ticks starting from 1
num_layers = array_transposed.shape[1]
num_experts = array_transposed.shape[0]
plt.xticks(ticks=np.arange(num_layers), labels=np.arange(1, num_layers + 1))
plt.yticks(ticks=np.arange(num_experts), labels=np.arange(1, num_experts + 1))

# Add labels and title
plt.xlabel('Layer')
plt.ylabel('Expert')
plt.title('Expert Distance')

# Save the heatmap
plt.savefig('expert_dist_heat_map.png', dpi=300)

