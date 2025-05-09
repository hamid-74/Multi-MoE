import matplotlib.pyplot as plt
import numpy as np

# X-axis labels in new order
labels = [
    'Model A', 'Model B', 'Model C', 'Model D',
    'Avg(A, B)', 'Avg(A, B, C)', 'Avg(A, B, C, D)',
    'Proposed(A, B)', 'Proposed(A, B, C)', 'Proposed(A, B, C, D)'
]

# Corresponding ROUGE-1 scores
rouge1 = [
    0.4939, 0.1487, 0.0523, 0.0404,
    0.4217, 0.3335, 0.2561,
    0.4911, 0.4600, 0.4633
]

plt.rcParams['font.family'] = 'serif'

# Create bar chart
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(14, 5.5))

# Plot bars with hatching
# bars = ax.bar(x, rouge1, color='white', edgecolor='black', hatch='//')
bars = ax.bar(x, rouge1, color='gainsboro', edgecolor='black')

# Axis formatting
ax.set_ylabel('ROUGE-1 Score')
# ax.set_title('ROUGE-1 Scores on SAMSum Dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')

# Add grid to y-axis
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('scalability.png', dpi=300)
