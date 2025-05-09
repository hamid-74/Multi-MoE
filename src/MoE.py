import copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

import random

import torch.nn as nn
import tqdm

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

class MoE:
    def __init__(self, args, tokenizer):
       
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        
        


        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            # device_map='cpu',
            use_cache=True,
        )
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        self.total_available_memory = args.total_available_memory

        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0


        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)



        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0


        self.bring_non_expert_to_gpu()

        print(f"size of non expert layers: {torch.cuda.memory_allocated()}")

        if self.total_available_memory:
            self.total_available_memory = self.total_available_memory - torch.cuda.memory_allocated()
        else:    
            self.total_available_memory = torch.cuda.get_device_properties(self.dev).total_memory * 0.95 - torch.cuda.memory_allocated(self.dev)
        
        # 0: CPU, 1: GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.n_experts_on_gpu = self.calc_n_experts_on_gpu()
        print(
            f"Number of experts on GPU: {self.n_experts_on_gpu}/{self.n_layer * self.n_expert}"
        )

        self.set_expert_loc(self.n_experts_on_gpu)
        # print(self.expert_loc)

        self.bring_expert_to_gpu()



        self.generated_logits = []

        print("Model is ready.")

    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self.model.layers[i].block_sparse_moe.gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            # only model.layers[i].block_sparse_moe.experts is on CPU

    def set_expert_loc(self, n_experts_on_gpu):
        """Set the location of experts"""

        all_expert_positions = [(i, j) for i in range(self.n_layer) for j in range(self.n_expert)]
        gpu_expert_positions = random.sample(all_expert_positions, self.n_experts_on_gpu)


    
        for i_layer, i_expert in gpu_expert_positions:
            self.expert_loc[i_layer, i_expert] = 1


    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)

    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1

    def calc_n_experts_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )

        print(f"expert total number of parameters: {n_param}")                
        return int((self.total_available_memory) // (n_param * 2))

    def generate(self, text, output_token=20, input_token=None, input_ids_flag=False, print_flag=False, temperature=1, do_sample=False
    ):
        self.generated_logits.clear()

        outputs = torch.empty((1, 0), dtype=torch.int64)
        
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        if(input_ids_flag):
            input_ids, position_ids = self.tokenize_input_ids(text)
        else:
            input_ids, position_ids = self.tokenize(text)

        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]

        tick = time.time()
        is_decode = False
        prefill_time, decode_time = 0, 0
        for i_token in range(output_token):
            # tick = time.time()
            if print_flag:
                print(self.tokenizer.decode(input_ids[0, :]))

            

            
            logits = self.mixtral_forward(
                input_ids,
                position_ids,
                is_decode,
            )
            

            # print('Time:', time.time() - tick)

            logits = logits.to("cpu")

            

            if do_sample and is_decode:
                logits = logits[:, -1, :]
                if temperature != 1.0:
                    logits = logits / temperature
                # Apply softmax to get probabilities
                
                probs = F.softmax(logits, dim=-1)
    
                # Sample from the distribution
                output = torch.multinomial(probs, num_samples=1)
            else:
                # Get the most likely token (argmax)
                output = torch.argmax(logits, dim=-1)
                
            outputs = torch.cat((outputs, output), dim=1)
            # print(f"output:{output}, outputs:{outputs}, dtype:{output.dtype}")


            self.past_key_values_length += output.shape[-1]
            input_ids = output[:, -1].unsqueeze(0).to(self.dev)
            
            position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + 1,
                dtype=torch.long,
                device=self.dev,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, 1)
            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True

            
            if(output[0][0]==2):
                print("eos found!!")
                break

        
        decode_time = time.time() - tick

        return prefill_time, decode_time, self.cnt_expert_hit, outputs

    def tokenize(self, text):
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.dev)
        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])
        return input_ids, position_ids
    
    def tokenize_input_ids(self, input_ids):

        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])
        return input_ids, position_ids

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, is_decode):
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.dev)
        inps = self.model.embed_tokens(inps)

        for i_layer, layer in enumerate(self.model.layers):
            # if i_layer == 31:
            #     continue
            original_inps_shape = inps.shape

            inps_residual = inps
            inps = layer.input_layernorm(inps)
            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)

            inps = inps.view(-1, hidden_dim)
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1)
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # intermediate variable to store the output of experts
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts


            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=8
            ).permute(2, 1, 0)

            for i_expert in range(len(experts)):
                is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                idx, top_2 = torch.where(expert_mask[i_expert])

                if top_2.shape[0] == 0:
                    # print(f"Expert {i_expert}: has no tokens")
                    continue

                # torch.cuda.synchronize()
                top_2_list = top_2.tolist()
                idx_list = idx.tolist()

                current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                if not is_cuda:
                    self.expert_placeholder.load_state_dict(
                        experts[i_expert].state_dict()
                    )
                    current_state = self.expert_placeholder(
                        current_state, routing_weights[top_2_list, idx_list, None]
                    )
                else:
                    current_state = experts[i_expert](
                        current_state, routing_weights[top_2_list, idx_list, None]
                    )
                inps_after_experts.index_add_(
                    0, top_2, current_state.to(inps.dtype)
                )

                if not is_cuda:
                    experts[i_expert] = experts[i_expert].to(
                        "cpu", non_blocking=True
                    )

                # end of one expert

            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

            # end of one layer

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis

