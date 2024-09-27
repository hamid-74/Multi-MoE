import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

def LayoutPlotter(model_layout, file_name):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Colors for each unique model
    model_colors = {
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "royalblue",
        "mistralai/Mixtral-8x7B-v0.1": "darkorange"
    }

    # Constants for drawing
    rect_height = 0.8
    
    rect_width = 0.35
    layer_spacing = 0.5
    expert_spacing = 0.1

    rect_height_expert = 0.6

    rect_height_attention = 8 * rect_height_expert + 7 * expert_spacing

    # Iterate through each layer
    for i in range(32):
        y_position = 0

        # Draw the attention layer
        ax.add_patch(patches.Rectangle((((i*2)-1) * layer_spacing, y_position), rect_width, rect_height_attention, 
                                       edgecolor='black', facecolor=model_colors[model_layout["non_expert"]]))

        # Draw the expert layers

        for j in range(8):
            ax.add_patch(patches.Rectangle(((i * layer_spacing)*2, y_position + j * (rect_height_expert + expert_spacing)), rect_width, rect_height_expert, 
                                           edgecolor='black', facecolor=model_colors[model_layout[f"expert_layer_{i}"]]))

    # Set axis limits
    ax.set_xlim(-0.5, 70 * layer_spacing)
    ax.set_ylim(rect_height - 1, 8 * (rect_height + expert_spacing) + 0.5)
    
    # Hide axes
    ax.axis('off')

    # Add legend
    handles = [patches.Patch(color=color, label=name) for name, color in model_colors.items()]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    plt.savefig(file_name, dpi=300)

