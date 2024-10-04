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


# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
# from mistral_common.protocol.instruct.messages import UserMessage
# from mistral_common.protocol.instruct.request import ChatCompletionRequest
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


model_layout = {}
model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
model_layout["non_expert"] = "mistralai/Mixtral-8x7B-v0.1"


for i in range(0,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
      
    

for i in range(0,32,2):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-v0.1"


args = make_args()

# # # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)


multimoe.generate("where is Yazd?", output_token = 50, print_flag = True, record_stats=False)
# _,exec_stats, _ = 

# print(exec_stats)