from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.colors import ListedColormap, BoundaryNorm

# Load the data from a JSON file
with open("expert_map.json", "r") as f:
    data = np.array(json.load(f))

# Transpose the data to swap rows and columns
data = data.T

# Create a colormap: -1 is blank, 0 is one color, 1 is another color
custom_colors = ['lightgray', 'forestgreen', 'darkorange']  # Visible blank points and strong colors
cmap = ListedColormap(custom_colors)

bounds = [-1.5, -0.5, 0.5, 1.5]       # Boundaries for the 3 values
norm = colors.BoundaryNorm(bounds, cmap.N)

# Mask the -1 values to make them blank
masked_data = np.ma.masked_where(data == -1, data)

# Configure matplotlib to use serif font
plt.rcParams['font.family'] = 'serif'

# Create the plot
fig, ax = plt.subplots(figsize=(16, 8))  # Adjust figsize for the new orientation
cbar = ax.imshow(masked_data, cmap=cmap, norm=norm, interpolation='none')

# Adjust axes ticks to start from 1 instead of 0
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))
ax.set_xticklabels(np.arange(1, data.shape[1] + 1))  # Start from 1
ax.set_yticklabels(np.arange(1, data.shape[0] + 1))  # Start from 1

# Add colorbar, skipping the masked value
plt.colorbar(cbar, ax=ax, boundaries=bounds, ticks=[0, 1])

# Set axis labels and title
ax.set_title("Expert Map", fontsize=24)
ax.set_xlabel("Layer", fontsize=24)
ax.set_ylabel("Expert", fontsize=24)

# Rotate x-axis labels if needed for better visibility
plt.xticks(rotation=45)

# Display the plot
plt.savefig('expert_map.png', dpi=300)
plt.show()
