import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM

# Load your two models
local_path_1 = "./switch-emre"
local_path_2 = "./switch-base"
local_path_3 = "./switch-sst2"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
model1 = AutoModelForSeq2SeqLM.from_pretrained(local_path_1).to(device)
model2 = AutoModelForSeq2SeqLM.from_pretrained(local_path_2).to(device)
model3 = AutoModelForSeq2SeqLM.from_pretrained(local_path_3).to(device)

def flatten_weights(weights):
    # Flatten a tensor into 1D
    return weights.view(-1)

def euclidean_distance(tensor1, tensor2):
    # Compute Euclidean distance between two tensors
    return torch.norm(tensor1 - tensor2)

def calculate_expert_distance(model1_state_dict, model2_state_dict, model3_state_dict):
    expert_keys = [
        'encoder.block.{block_num}.layer.{layer_num}.mlp.experts.expert_{expert_num}.wi.weight',
        'encoder.block.{block_num}.layer.{layer_num}.mlp.experts.expert_{expert_num}.wo.weight',
        'decoder.block.{block_num}.layer.{layer_num}.mlp.experts.expert_{expert_num}.wi.weight',
        'decoder.block.{block_num}.layer.{layer_num}.mlp.experts.expert_{expert_num}.wo.weight'
    ]
    
    encoder_distances = np.zeros((6, 8))  # Initialize a 6x8 array for the encoder (for 12 blocks and 8 experts)
    decoder_distances = np.zeros((6, 8))  # Initialize a 6x8 array for the decoder (for 12 blocks and 8 experts)
    num_experts = 8  # Assuming 8 experts per layer
    
    # Calculate distances for encoder
    for block_num in range(1, 13, 2):  # Assuming 12 blocks in the encoder
        for layer_num in range(1, 2):  # Assuming each block has 1 layer
            for expert_num in range(num_experts):
                # model 1 and 2
                # Build the key for wi and wo weights for the encoder
                wi_key = expert_keys[0].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)
                wo_key = expert_keys[1].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)

                if wi_key in model1_state_dict and wi_key in model2_state_dict:
                    # Flatten wi weights
                    wi_tensor_1 = flatten_weights(model1_state_dict[wi_key])
                    wi_tensor_2 = flatten_weights(model2_state_dict[wi_key])
                else:
                    print("error encode")
                    print(wi_key)
                    continue

                if wo_key in model1_state_dict and wo_key in model2_state_dict:
                    # Flatten wo weights
                    wo_tensor_1 = flatten_weights(model1_state_dict[wo_key])
                    wo_tensor_2 = flatten_weights(model2_state_dict[wo_key])
                else:
                    print("error encode")
                    print(wo_key)
                    continue

                # Concatenate wi and wo into a single 1D tensor
                concatenated_tensor_1 = torch.cat((wi_tensor_1, wo_tensor_1))
                concatenated_tensor_2 = torch.cat((wi_tensor_2, wo_tensor_2))

                # Compute Euclidean distance between concatenated tensors
                distance = euclidean_distance(concatenated_tensor_1, concatenated_tensor_2)
                encoder_distances[block_num // 2, expert_num] += distance  # Store the distance in the encoder array



                # model 1 and 3
                # Build the key for wi and wo weights for the encoder
                wi_key = expert_keys[0].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)
                wo_key = expert_keys[1].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)

                if wi_key in model1_state_dict and wi_key in model3_state_dict:
                    # Flatten wi weights
                    wi_tensor_1 = flatten_weights(model1_state_dict[wi_key])
                    wi_tensor_2 = flatten_weights(model3_state_dict[wi_key])
                else:
                    print("error encode")
                    print(wi_key)
                    continue

                if wo_key in model1_state_dict and wo_key in model3_state_dict:
                    # Flatten wo weights
                    wo_tensor_1 = flatten_weights(model1_state_dict[wo_key])
                    wo_tensor_2 = flatten_weights(model3_state_dict[wo_key])
                else:
                    print("error encode")
                    print(wo_key)
                    continue

                # Concatenate wi and wo into a single 1D tensor
                concatenated_tensor_1 = torch.cat((wi_tensor_1, wo_tensor_1))
                concatenated_tensor_2 = torch.cat((wi_tensor_2, wo_tensor_2))

                # Compute Euclidean distance between concatenated tensors
                distance = euclidean_distance(concatenated_tensor_1, concatenated_tensor_2)
                encoder_distances[block_num // 2, expert_num] += distance  # Store the distance in the encoder array

                # model 2 and 3
                # Build the key for wi and wo weights for the encoder
                wi_key = expert_keys[0].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)
                wo_key = expert_keys[1].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)

                if wi_key in model2_state_dict and wi_key in model3_state_dict:
                    # Flatten wi weights
                    wi_tensor_1 = flatten_weights(model2_state_dict[wi_key])
                    wi_tensor_2 = flatten_weights(model3_state_dict[wi_key])
                else:
                    print("error encode")
                    print(wi_key)
                    continue

                if wo_key in model2_state_dict and wo_key in model3_state_dict:
                    # Flatten wo weights
                    wo_tensor_1 = flatten_weights(model2_state_dict[wo_key])
                    wo_tensor_2 = flatten_weights(model3_state_dict[wo_key])
                else:
                    print("error encode")
                    print(wo_key)
                    continue

                # Concatenate wi and wo into a single 1D tensor
                concatenated_tensor_1 = torch.cat((wi_tensor_1, wo_tensor_1))
                concatenated_tensor_2 = torch.cat((wi_tensor_2, wo_tensor_2))

                # Compute Euclidean distance between concatenated tensors
                distance = euclidean_distance(concatenated_tensor_1, concatenated_tensor_2)
                encoder_distances[block_num // 2, expert_num] += distance  # Store the distance in the encoder array


    # Calculate distances for decoder
    for block_num in range(1, 13, 2):  # Assuming 12 blocks in the decoder
        for layer_num in range(2, 3):  # Assuming each block has 1 layer
            for expert_num in range(num_experts):
                # model 1 and 2
                # Build the key for wi and wo weights for the decoder
                wi_key = expert_keys[2].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)
                wo_key = expert_keys[3].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)

                if wi_key in model1_state_dict and wi_key in model2_state_dict:
                    # Flatten wi weights
                    wi_tensor_1 = flatten_weights(model1_state_dict[wi_key])
                    wi_tensor_2 = flatten_weights(model2_state_dict[wi_key])
                else:
                    print("error decode")
                    print(wi_key)
                    continue

                if wo_key in model1_state_dict and wo_key in model2_state_dict:
                    # Flatten wo weights
                    wo_tensor_1 = flatten_weights(model1_state_dict[wo_key])
                    wo_tensor_2 = flatten_weights(model2_state_dict[wo_key])
                else:
                    print("error decode")
                    print(wo_key)
                    continue

                # Concatenate wi and wo into a single 1D tensor
                concatenated_tensor_1 = torch.cat((wi_tensor_1, wo_tensor_1))
                concatenated_tensor_2 = torch.cat((wi_tensor_2, wo_tensor_2))

                # Compute Euclidean distance between concatenated tensors
                distance = euclidean_distance(concatenated_tensor_1, concatenated_tensor_2)
                decoder_distances[block_num // 2, expert_num] += distance  # Store the distance in the decoder array
                
                
                # model 1 and 3
                # Build the key for wi and wo weights for the decoder
                wi_key = expert_keys[2].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)
                wo_key = expert_keys[3].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)

                if wi_key in model1_state_dict and wi_key in model3_state_dict:
                    # Flatten wi weights
                    wi_tensor_1 = flatten_weights(model1_state_dict[wi_key])
                    wi_tensor_2 = flatten_weights(model3_state_dict[wi_key])
                else:
                    print("error decode")
                    print(wi_key)
                    continue

                if wo_key in model1_state_dict and wo_key in model3_state_dict:
                    # Flatten wo weights
                    wo_tensor_1 = flatten_weights(model1_state_dict[wo_key])
                    wo_tensor_2 = flatten_weights(model3_state_dict[wo_key])
                else:
                    print("error decode")
                    print(wo_key)
                    continue

                # Concatenate wi and wo into a single 1D tensor
                concatenated_tensor_1 = torch.cat((wi_tensor_1, wo_tensor_1))
                concatenated_tensor_2 = torch.cat((wi_tensor_2, wo_tensor_2))

                # Compute Euclidean distance between concatenated tensors
                distance = euclidean_distance(concatenated_tensor_1, concatenated_tensor_2)
                decoder_distances[block_num // 2, expert_num] += distance  # Store the distance in the decoder array


                # model 2 and 3
                # Build the key for wi and wo weights for the decoder
                wi_key = expert_keys[2].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)
                wo_key = expert_keys[3].format(block_num=block_num, layer_num=layer_num, expert_num=expert_num)

                if wi_key in model2_state_dict and wi_key in model3_state_dict:
                    # Flatten wi weights
                    wi_tensor_1 = flatten_weights(model2_state_dict[wi_key])
                    wi_tensor_2 = flatten_weights(model3_state_dict[wi_key])
                else:
                    print("error decode")
                    print(wi_key)
                    continue

                if wo_key in model2_state_dict and wo_key in model3_state_dict:
                    # Flatten wo weights
                    wo_tensor_1 = flatten_weights(model2_state_dict[wo_key])
                    wo_tensor_2 = flatten_weights(model3_state_dict[wo_key])
                else:
                    print("error decode")
                    print(wo_key)
                    continue

                # Concatenate wi and wo into a single 1D tensor
                concatenated_tensor_1 = torch.cat((wi_tensor_1, wo_tensor_1))
                concatenated_tensor_2 = torch.cat((wi_tensor_2, wo_tensor_2))

                # Compute Euclidean distance between concatenated tensors
                distance = euclidean_distance(concatenated_tensor_1, concatenated_tensor_2)
                decoder_distances[block_num // 2, expert_num] += distance  # Store the distance in the decoder array

    return encoder_distances, decoder_distances

# Assuming model1_state_dict and model2_state_dict are the state dicts of your models
model1_state_dict = model1.state_dict()  # Model 1 state dict
model2_state_dict = model2.state_dict()  # Model 2 state dict
model3_state_dict = model3.state_dict()  # Model 2 state dict

# Calculate the distances
encoder_distances, decoder_distances = calculate_expert_distance(model1_state_dict, model2_state_dict, model3_state_dict)


# Print the encoder distances in a 6x8 format
print("Encoder Expert Distances:")
for row in encoder_distances:
    print(row)

# Print the decoder distances in a 6x8 format
print("\nDecoder Expert Distances:")
for row in decoder_distances:
    print(row)

encoder_dict = {}
decoder_dict = {}

for i in range(6):
    for j in range(8):
        # 1 is the label for encoder and 2 is the label for decoder
        encoder_dict[(i, j, 1)] = encoder_distances[i][j]
        decoder_dict[(i, j, 2)] = decoder_distances[i][j]


combined_dict = list(encoder_dict.items()) + list(decoder_dict.items())

# Sort the list based on the values
sorted_combined = sorted(combined_dict, key=lambda x: x[1])

# Extract the sorted keys and values
sorted_keys = [item[0] for item in sorted_combined]
sorted_values = [item[1] for item in sorted_combined]

# Print the sorted keys and values
print("Sorted Keys:")
print(sorted_keys)
print("\nSorted Values:")
print(sorted_values)



def merge_models(model1_state_dict, model2_state_dict, model3_state_dict, sorted_keys, sorted_values):
    merged_model_state_dict = model1_state_dict.copy()
    

    for i in range(len(sorted_keys)):

        if i > 96:
            break
        
        block_num = sorted_keys[i][0] * 2 + 1

        expert_num = sorted_keys[i][1]  # Expert number

        if sorted_keys[i][2] == 1:
            encdec = "encoder"
            layer_num = 1
        elif sorted_keys[i][2] == 2:
            encdec = "decoder"
            layer_num = 2
        else:
            print("error encdec")
        
        # Determine which model's expert to pick based on round-robin strategy
        if i % 3 == 0:
            model_num = 1  # From model 1
        elif i % 3 == 1:
            model_num = 2  # From model 2
        elif i % 3 == 2:
            model_num = 3 # From model 3
        else:
            print("error iterator")
        
        # Build the keys for wi and wo
        wi_key = f'{encdec}.block.{block_num}.layer.{layer_num}.mlp.experts.expert_{expert_num}.wi.weight'
        wo_key = f'{encdec}.block.{block_num}.layer.{layer_num}.mlp.experts.expert_{expert_num}.wo.weight'
        
        if model_num == 1:
            merged_model_state_dict[wi_key] = model1_state_dict[wi_key]
            merged_model_state_dict[wo_key] = model1_state_dict[wo_key]
        elif model_num == 2:
            merged_model_state_dict[wi_key] = model2_state_dict[wi_key]
            merged_model_state_dict[wo_key] = model2_state_dict[wo_key]
        elif model_num == 3:
            merged_model_state_dict[wi_key] = model3_state_dict[wi_key]
            merged_model_state_dict[wo_key] = model3_state_dict[wo_key]

    
    return merged_model_state_dict

# Merge the experts
merged_model_base_state_dict = merge_models(model1_state_dict, model2_state_dict, model3_state_dict, sorted_keys, sorted_values)


merged_model = AutoModelForSeq2SeqLM.from_pretrained(local_path_1)  # Load one of the models as a base
merged_model.load_state_dict(merged_model_base_state_dict)  # Load the merged state dict

# Save the merged model
merged_model.save_pretrained("merged-emre-base-sst2")

print("Merged model saved in 'merged-emre-base-sst2' folder.")


# # Merge the experts
# merged_model_emre_state_dict = merge_models_emre(model1_state_dict, model2_state_dict, sorted_keys, sorted_values)


# merged_model_emre = AutoModelForSeq2SeqLM.from_pretrained(local_path_2)  # Load one of the models as a base
# merged_model_emre.load_state_dict(merged_model_emre_state_dict)  # Load the merged state dict

# # Save the merged model
# merged_model_emre.save_pretrained("merged-sst2")

# print("Merged model saved in 'merged-emre' folder.")