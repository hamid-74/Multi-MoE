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

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pickle

dataset_labels = ['wikitext', 'ptb_text_only', 'c4']
dataset_samples = 128


class Evaluator:
    def __init__(self, dataset, column, tokenizer, device, n_samples=dataset_samples):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset[column]), return_tensors="pt"
        ).input_ids

        # print(f"dataset shape:{self.dataset.shape}")
        self.n_samples = n_samples


    


    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            # Get input sequence for encoder
            batch = self.dataset[:, (i * 512) : ((i + 1) * 512)].to(model.device)
            # print(f"batch shape: {batch.shape}")

            # Create decoder_input_ids by shifting the target sequence
            decoder_input_ids = batch[:, :-1].contiguous()  # Shift the target sequence by 1 for decoder input

            # Pass both input_ids and decoder_input_ids to the model
            outputs = model(input_ids=batch, decoder_input_ids=decoder_input_ids)
            
            # Get logits from the decoder
            lm_logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]
            # print(f"lm_logits shape: {lm_logits.shape}")

            # Shift logits and labels for the loss computation
            shift_logits = lm_logits[:, :-1, :].contiguous().float()  # Ignore the last token
            shift_labels = batch[:, 1:].contiguous()  # Shift the labels by one position

            # Now, slice shift_labels to match the size of shift_logits (both should have shape [batch_size, seq_length - 1])
            shift_labels = shift_labels[:, :-1]  # Slice to remove the last token for matching shapes

            # Calculate loss for each token
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            
            # Negative log-likelihood
            neg_log_likelihood = loss.float() * 512  # Multiply by 512 to account for batch size
            nlls.append(neg_log_likelihood)

        # Return perplexity (exp of the average NLL)
        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 512))





# Load from local directory
local_path = "./switch-sst2"
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_path).to("cuda")




ppl_dict = dict()




for label in dataset_labels: 
    if label == 'wikitext':
        dataset = load_dataset('../wikitext', split='test')

    elif label == 'ptb_text_only':
        dataset = load_dataset('../ptb_text_only', 'default', split='train')

    elif label == 'c4':
        dataset = load_dataset('../allenai_c4', split='validation')
        dataset = dataset.select(range(600))


    if label == 'wikitext' or label == 'c4':
        evaluator = Evaluator(dataset, "text", tokenizer, "cuda")
    elif label == 'ptb_text_only':
        evaluator = Evaluator(dataset, "sentence", tokenizer, "cuda")


    ppl = evaluator.evaluate(model)
    ppl_dict[label] = float(ppl)
    print(f"perplexity of dataset {label}: {ppl}")

    








