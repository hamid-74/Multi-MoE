import json
import matplotlib.pyplot as plt


# Load the data from the JSON file
with open('results/gpu_offloading_throughput.json', 'r') as file:
    data = json.load(file)

# Initialize lists for each no1 value
no1_16 = []
no1_32 = []
no1_64 = []
no1_128 = []

# Iterate through the keys and store values in respective lists
for key in data.keys():
    no1_value, no2_value = map(int, key.split('_'))
    if no1_value == 16:
        no1_16.append(data[key])
    elif no1_value == 32:
        no1_32.append(data[key])
    elif no1_value == 64:
        no1_64.append(data[key])
    elif no1_value == 128:
        no1_128.append(data[key])

# Print or do whatever you want with the lists
# print("Values for no1=16:", no1_16)
# print("Values for no1=32:", no1_32)
# print("Values for no1=64:", no1_64)
# print("Values for no1=128:", no1_128)




fig, axs = plt.subplots(4, 1, figsize=(10, 20))

# Plot for no1=16
axs[0].bar(range(len(no1_16)), no1_16)
axs[0].set_title('no1 = 16')
axs[0].set_xticks(range(len(no1_16)))
axs[0].set_xticklabels(['no2=' + str(val) for val in [16, 32, 64, 128, 256, 512]])

# Plot for no1=32
axs[1].bar(range(len(no1_32)), no1_32)
axs[1].set_title('no1 = 32')
axs[1].set_xticks(range(len(no1_32)))
axs[1].set_xticklabels(['no2=' + str(val) for val in [16, 32, 64, 128, 256, 512]])

# Plot for no1=64
axs[2].bar(range(len(no1_64)), no1_64)
axs[2].set_title('no1 = 64')
axs[2].set_xticks(range(len(no1_64)))
axs[2].set_xticklabels(['no2=' + str(val) for val in [16, 32, 64, 128, 256, 512]])

# Plot for no1=128
axs[3].bar(range(len(no1_128)), no1_128)
axs[3].set_title('no1 = 128')
axs[3].set_xticks(range(len(no1_128)))
axs[3].set_xticklabels(['no2=' + str(val) for val in [16, 32, 64, 128, 256, 512]])

plt.tight_layout()
plt.savefig('plots/bar_chart.png')