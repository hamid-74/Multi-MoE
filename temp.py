import copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

import torch.nn as nn
from tqdm import tqdm
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





# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", load_in_8bit=True)
# print(model)

# text = "where is Yazd?"
# inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

# print(f"{inputs}")

# outputs = model.generate(**inputs, max_new_tokens=50)
# print(outputs)
# print(outputs.shape)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))



model_layout = {}
model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
model_layout["non_expert"] = "mistralai/Mixtral-8x7B-v0.1"




args = make_args()

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)



_, _, output_ids = multimoe.generate("Where is washington dc?", output_token = 50, print_flag = True, record_stats=False)


print(output_ids[0])
print(output_ids.shape)
print(tokenizer.decode(output_ids[0]))
