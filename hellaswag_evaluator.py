"""
CREDITS: This source file has been adopted as is from -
https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py 

Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
import argparse
import os
import tqdm
import shortuuid
import random
from datasets import load_dataset
import json
import torch.nn as nn
import transformers
import torch.autograd.profiler as profiler
import threading
import time
from src.MultiMoE import MultiMoE
from src.CustomEvaluator import CustomEvaluator
from src.LayoutPlotter import LayoutPlotter
import pickle
import requests

from tqdm import tqdm


DATA_CACHE_DIR = "hellaswag"





def make_args():
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )

    parser.add_argument(
        "--total-available-memory",
        type=int,
        default=83900000000,
        help="total gpu memory",
    )

    args = parser.parse_args()
    return args


def render_example(example, tokenizer):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = tokenizer.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = tokenizer.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
        end_tokens = end_tokens[1:]
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split):
    file_path = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    
    # Load all lines from the file
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Load sampled indexes from file
    with open("hellaswag/sampled_indexes.json", "r") as f:
        sampled_indexes = json.load(f)
    
    # Sample the lines using saved indexes
    sampled_lines = [lines[i] for i in sampled_indexes]
    
    # Yield the examples from the sampled lines
    for line in sampled_lines:
        example = json.loads(line)
        yield example


@torch.no_grad()
def evaluate(model, tokenizer, device):

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    file_path = os.path.join(DATA_CACHE_DIR, f"hellaswag_val.jsonl")
    
    # Load all lines from the file
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Load sampled indexes from file
    with open("hellaswag/sampled_indexes.json", "r") as f:
        sampled_indexes = json.load(f)
    
    # Sample the lines using saved indexes
    sampled_lines = [lines[i] for i in sampled_indexes]

    print(f"len samples lines: {len(sampled_lines)}")

    for line in sampled_lines:
        example = json.loads(line)
        data, tokens, mask, label = render_example(example, tokenizer)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits_list = []
        

        for i in range(len(tokens)):
            with torch.no_grad():
            # lm_logits = model(batch).logits
                tokens_reshaped = tokens[i].unsqueeze(0)
                generated_outputs = model.generate(tokens_reshaped, output_token=1, input_ids_flag=True, ppl_flag=True)

            logits_list.append(generated_outputs[0][0][0])
        
        

        logits = torch.stack(logits_list)



        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

    return num_correct_norm/num_total





results_folder = "/multi-llm/fiddler/fiddler/results/hellaswag/test_1/"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

model_layout = {}

model_layout["non_expert"] = "mistralai/Mixtral-8x7B-v0.1"


for i in range(0,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-v0.1"
    
    

# for i in range(16,32):
#     key = "expert_layer_" + str(i)
#     model_layout[key] = "mistralai/Mixtral-8x7B-Instruct-v0.1"

if not os.path.exists(results_folder):
    # Create the folder if it doesn't exist
    os.makedirs(results_folder)


layout_file = os.path.join(results_folder, 'layout.png')
LayoutPlotter(model_layout, layout_file)

layout_file = os.path.join(results_folder, 'model_layout.json')
with open(layout_file, "w") as json_file:
    json.dump(model_layout, json_file)

args = make_args()

# # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)


score = evaluate(multimoe, tokenizer, torch.device("cuda:0"))


score_file = os.path.join(results_folder, 'score.json')
with open(score_file, "w") as json_file:
    json.dump(score, json_file)


del(multimoe)
torch.cuda.empty_cache()




results_folder = "/multi-llm/fiddler/fiddler/results/hellaswag/test_2/"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

model_layout = {}

model_layout["non_expert"] = "mistralai/Mixtral-8x7B-v0.1"


for i in range(0,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-v0.1"
    
    

for i in range(16,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-Instruct-v0.1"

if not os.path.exists(results_folder):
    # Create the folder if it doesn't exist
    os.makedirs(results_folder)


layout_file = os.path.join(results_folder, 'layout.png')
LayoutPlotter(model_layout, layout_file)

layout_file = os.path.join(results_folder, 'model_layout.json')
with open(layout_file, "w") as json_file:
    json.dump(model_layout, json_file)

args = make_args()

# # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)


score = evaluate(multimoe, tokenizer, torch.device("cuda:0"))


score_file = os.path.join(results_folder, 'score.json')
with open(score_file, "w") as json_file:
    json.dump(score, json_file)


del(multimoe)
torch.cuda.empty_cache()



results_folder = "/multi-llm/fiddler/fiddler/results/hellaswag/test_3/"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

model_layout = {}

model_layout["non_expert"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"


for i in range(0,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    

# for i in range(16,32):
#     key = "expert_layer_" + str(i)
#     model_layout[key] = "mistralai/Mixtral-8x7B-Instruct-v0.1"

if not os.path.exists(results_folder):
    # Create the folder if it doesn't exist
    os.makedirs(results_folder)


layout_file = os.path.join(results_folder, 'layout.png')
LayoutPlotter(model_layout, layout_file)

layout_file = os.path.join(results_folder, 'model_layout.json')
with open(layout_file, "w") as json_file:
    json.dump(model_layout, json_file)

args = make_args()

# # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)


score = evaluate(multimoe, tokenizer, torch.device("cuda:0"))


score_file = os.path.join(results_folder, 'score.json')
with open(score_file, "w") as json_file:
    json.dump(score, json_file)


del(multimoe)
torch.cuda.empty_cache()






results_folder = "/multi-llm/fiddler/fiddler/results/hellaswag/test_4/"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

model_layout = {}

model_layout["non_expert"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"


for i in range(0,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    
    

for i in range(16,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-v0.1"
    

if not os.path.exists(results_folder):
    # Create the folder if it doesn't exist
    os.makedirs(results_folder)


layout_file = os.path.join(results_folder, 'layout.png')
LayoutPlotter(model_layout, layout_file)

layout_file = os.path.join(results_folder, 'model_layout.json')
with open(layout_file, "w") as json_file:
    json.dump(model_layout, json_file)

args = make_args()

# # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)


score = evaluate(multimoe, tokenizer, torch.device("cuda:0"))


score_file = os.path.join(results_folder, 'score.json')
with open(score_file, "w") as json_file:
    json.dump(score, json_file)





del(multimoe)
torch.cuda.empty_cache()






results_folder = "/multi-llm/fiddler/fiddler/results/hellaswag/test_5/"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

model_layout = {}

model_layout["non_expert"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"


for i in range(0,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-v0.1"
    
    
    
    

for i in range(16,32):
    key = "expert_layer_" + str(i)
    
    model_layout[key] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    

if not os.path.exists(results_folder):
    # Create the folder if it doesn't exist
    os.makedirs(results_folder)


layout_file = os.path.join(results_folder, 'layout.png')
LayoutPlotter(model_layout, layout_file)

layout_file = os.path.join(results_folder, 'model_layout.json')
with open(layout_file, "w") as json_file:
    json.dump(model_layout, json_file)

args = make_args()

# # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)


score = evaluate(multimoe, tokenizer, torch.device("cuda:0"))


score_file = os.path.join(results_folder, 'score.json')
with open(score_file, "w") as json_file:
    json.dump(score, json_file)





del(multimoe)
torch.cuda.empty_cache()






results_folder = "/multi-llm/fiddler/fiddler/results/hellaswag/test_6/"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

model_layout = {}

model_layout["non_expert"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"


for i in range(0,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    
    

for i in range(0,32,2):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-v0.1"
    

if not os.path.exists(results_folder):
    # Create the folder if it doesn't exist
    os.makedirs(results_folder)


layout_file = os.path.join(results_folder, 'layout.png')
LayoutPlotter(model_layout, layout_file)

layout_file = os.path.join(results_folder, 'model_layout.json')
with open(layout_file, "w") as json_file:
    json.dump(model_layout, json_file)

args = make_args()

# # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)


score = evaluate(multimoe, tokenizer, torch.device("cuda:0"))


score_file = os.path.join(results_folder, 'score.json')
with open(score_file, "w") as json_file:
    json.dump(score, json_file)



