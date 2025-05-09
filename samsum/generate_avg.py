import torch
from transformers import AutoModelForSeq2SeqLM

# Load your two models
# local_path_1 = "./switch-base"
# local_path_2 = "./switch-emre"

local_path_1 = "./switch-sst2"
local_path_2 = "./switch-mnli"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
model_1 = AutoModelForSeq2SeqLM.from_pretrained(local_path_1).to(device)
model_2 = AutoModelForSeq2SeqLM.from_pretrained(local_path_2).to(device)

def average_models(model_1, model_2):
    # Create a new model to store averaged weights
    averaged_model = AutoModelForSeq2SeqLM.from_pretrained(local_path_1).to(device)  # Assuming both models have the same architecture
    
    # Average the shared embeddings explicitly
    if hasattr(model_1, 'shared') and hasattr(model_2, 'shared'):
        averaged_model.shared.weight.data.copy_(
            (model_1.shared.weight.data + model_2.shared.weight.data) / 2
        )
    else:
        print("Shared embeddings not found!")

    # Iterate through parameters and average them
    for param_name, param_1 in model_1.named_parameters():
        if param_name in dict(model_2.named_parameters()):  # Ensure the parameter exists in model_2
            param_2 = dict(model_2.named_parameters())[param_name]  # Get corresponding parameter in model_2
            
            # Average the weights
            averaged_param = (param_1.data + param_2.data) / 2
            
            # Now check if the parameter exists in averaged_model
            if param_name in averaged_model.state_dict():
                averaged_model.state_dict()[param_name].data.copy_(averaged_param)
            else:
                print(f"Warning: {param_name} not found in averaged model!")

    # Handle other buffers if necessary, such as LayerNorms
    for buffer_name, buffer_1 in model_1.named_buffers():
        if buffer_name in dict(model_2.named_buffers()):
            buffer_2 = dict(model_2.named_buffers())[buffer_name]
            averaged_buffer = (buffer_1.data + buffer_2.data) / 2
            if buffer_name in averaged_model.state_dict():
                averaged_model.state_dict()[buffer_name].data.copy_(averaged_buffer)
            else:
                print(f"Warning: {buffer_name} not found in averaged model!")

    return averaged_model

# Average the two models
averaged_model = average_models(model_1, model_2)

# Save the averaged model
averaged_model.save_pretrained("./switch-avg-sst2-mnli")

print("Averaged model saved to './switch-avg-sst2-mnli'")
