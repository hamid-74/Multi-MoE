import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

# Model name
model_name = "mistralai/Mixtral-8x7B-v0.1"

# Load the model in 4-bit precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,

)

# Load tokenizer (optional)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Count parameters for expert and non-expert layers
total_params = 0
expert_params = 0
non_expert_params = 0

for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    if "expert" in name:  # Customize this based on how experts are named in your model
        expert_params += num_params
    else:
        non_expert_params += num_params

print(f"Total parameters: {total_params / 1e6:.2f}M")
print(f"Expert parameters: {expert_params / 1e6:.2f}M")
print(f"Non-expert parameters: {non_expert_params / 1e6:.2f}M")
