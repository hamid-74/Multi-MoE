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


import pickle

class CustomEvaluator:
    def __init__(self, dataset, column, tokenizer, device, n_samples=128, general=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.general = general



        self.dataset = tokenizer(
            "\n\n".join(dataset[column]), return_tensors="pt"
        ).input_ids.to(device)


        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):

        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch_input_ids = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(self.device)

           
            if(not self.general):
                with torch.no_grad():

                    generated_outputs = model.generate(batch_input_ids, output_token=1, input_ids_flag=True, ppl_flag=True)
                lm_logits = generated_outputs[0]

                lm_logits = lm_logits[0]
            elif(self.general):
                with torch.no_grad():
                    lm_logits = model(batch_input_ids).logits

            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))

