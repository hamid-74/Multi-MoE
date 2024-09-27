import copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

import torch.nn as nn

import tqdm
import shortuuid

import random

from datasets import load_dataset
import json

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import torch.autograd.profiler as profiler


import copy
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from src.MultiMoE import MultiMoE

from src.CustomEvaluator import CustomEvaluator
from src.LayoutPlotter import LayoutPlotter

import pickle





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
        default=80900000000,
        help="total gpu memory",
    )

    args = parser.parse_args()
    return args

folder_name = "/multi-llm/fiddler/fiddler/results/ppl/test_4/"
model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

model_layout = {}

model_layout["non_expert"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"


for i in range(0,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
      
    

for i in range(0,32,2):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-v0.1"
    

## plot and save model layout in the folder
if not os.path.exists(folder_name):
    # Create the folder if it doesn't exist
    os.makedirs(folder_name)


layout_file = os.path.join(folder_name, 'layout.png')
LayoutPlotter(model_layout, layout_file)

args = make_args()

# # # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)


dataset_labels = ['wikitext', 'ptb_text_only', 'c4']
dataset_samples = 128
# # dataset_samples = 1



ppl_mm = dict()


for label in dataset_labels: 
    if label == 'wikitext':
        dataset = load_dataset('wikitext', split='test')

    elif label == 'ptb_text_only':
        dataset = load_dataset('ptb_text_only', 'default', split='train')

    elif label == 'c4':
        dataset = load_dataset('allenai_c4', split='validation')
        dataset = dataset.select(range(600))

    if label == 'wikitext' or label == 'c4':
        evaluator = CustomEvaluator(dataset, "text", tokenizer, "cuda", dataset_samples)
    elif label == 'ptb_text_only':
        evaluator = CustomEvaluator(dataset, "sentence", tokenizer, "cuda", dataset_samples)


    ppl = evaluator.evaluate(multimoe)
    ppl_mm[label] = float(ppl)
    print(f"perplexity of dataset {label}: {ppl}")

    
if not os.path.exists(folder_name):
    # Create the folder if it doesn't exist
    os.makedirs(folder_name)

ppl_file = os.path.join(folder_name, 'ppl.json')
layout_file = os.path.join(folder_name, 'model_layout.json')


# Dump the dictionary to the file
with open(ppl_file, "w") as json_file:
    json.dump(ppl_mm, json_file)

with open(layout_file, "w") as json_file:
    json.dump(model_layout, json_file)






# ppl_4bit = dict()

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", bnb_4bit_compute_dtype=torch.bfloat16, load_in_4bit=True)
# # model = FiddlerMixtral_4bit("mistralai/Mixtral-8x7B-v0.1")


# for label in dataset_labels: 
#     if label == 'wikitext':
#         dataset = load_dataset('wikitext', split='test')

#     elif label == 'ptb_text_only':
#         dataset = load_dataset('ptb_text_only', 'default', split='train')

#     elif label == 'c4':
#         dataset = load_dataset('allenai_c4', split='validation')
#         dataset = dataset.select(range(600))


#     if label == 'wikitext' or label == 'c4':
#         evaluator = CustomEvaluator(dataset, "text", tokenizer, "cuda", general=True)
#     elif label == 'ptb_text_only':
#         evaluator = CustomEvaluator(dataset, "sentence", tokenizer, "cuda", general=True)


#     ppl = evaluator.evaluate(model)
#     ppl_4bit[label] = float(ppl)
#     print(f"4bit model perplexity of dataset {label}: {ppl}")


# # file_path = "/fiddler/fiddler/results/4bit_ppl.json"

# # # Dump the dictionary to the file
# # with open(file_path, "w") as json_file:
# #     json.dump(ppl_4bit, json_file)