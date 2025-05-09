


import torch
from transformers import MixtralForCausalLM
import transformers

def average_moe_models(model1, model2):
    """Averages the weights of two MixtralForCausalLM models.

    Args:
        model1: The first MixtralForCausalLM model.
        model2: The second MixtralForCausalLM model.

    Returns:
        A new MixtralForCausalLM model with averaged weights, or None if the models' structures are incompatible.
        Raises a ValueError if the models are not of the correct type.
    """

    if not isinstance(model1, MixtralForCausalLM) or not isinstance(model2, MixtralForCausalLM):
        raise ValueError("Both inputs must be instances of MixtralForCausalLM.")

    model_avg = transformers.MixtralForCausalLM.from_pretrained(
                    "mistralai/Mixtral-8x7B-v0.1",
                    torch_dtype=torch.bfloat16,
                    use_cache=True,
                )

    model1_dict = model1.state_dict()
    model2_dict = model2.state_dict()
    model_avg_dict = model_avg.state_dict()

    for key in model1_dict:
        print(key)
        if key in model2_dict:
            try:
                 model_avg_dict[key] = (model1_dict[key] + model2_dict[key]) / 2
            except RuntimeError as e:
                print(f"Error averaging key {key}: {e}")
                print(f"Shape Model 1: {model1_dict[key].shape}")
                print(f"Shape Model 2: {model2_dict[key].shape}")
                return None # Return None if there's a shape mismatch
        else:
            print(f"Key {key} not found in both models. Skipping.")




    model_avg.load_state_dict(model_avg_dict)
    return model_avg


# Example usage (assuming you have loaded your models):
try:
        # Paths to your models and the output
    model_id1 = "mistralai/Mixtral-8x7B-v0.1"
    model_id2 = "mistralai/Mixtral-8x7B-Instruct-v0.1"


    # Load the first model
    model1 = transformers.MixtralForCausalLM.from_pretrained(
                    model_id1,
                    torch_dtype=torch.bfloat16,
                    use_cache=True,
                )
    # Load the first model
    model2 = transformers.MixtralForCausalLM.from_pretrained(
                    model_id2,
                    torch_dtype=torch.bfloat16,
                    use_cache=True,
                )
    averaged_model = average_moe_models(model1, model2)

    if averaged_model:
        print("Models averaged successfully.")
        # Save the averaged model
        averaged_model.save_pretrained("mistralai/Mixtral-avg")
    else:
        print("Models could not be averaged due to structural differences.")

except Exception as e:
    print(f"An error occurred: {e}")