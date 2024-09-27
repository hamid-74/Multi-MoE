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


from transformers.modeling_attn_mask_utils import *
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)


class ExecStats:
    def __init__(self, n_layer):
        self.embed = {'prefill': [], 'decode': []}
        
        self.layer = {'prefill': {}, 'decode': {}}
        for i in range(n_layer):
            self.layer['prefill'][f'layer_{i}'] = {
                'attention': [],
                'expert': {
                    'time': [],
                    'hit': []
                }
            }
            self.layer['decode'][f'layer_{i}'] = {
                'attention': [],
                'expert': {
                    'time': [],
                    'hit': []
                }
            }
        
        self.lm_head = {'prefill': [], 'decode': []}

    def __str__(self):
        # Formatting the output for better readability
        layer_str = ""
        for key in self.layer:
            layer_str += f"    {key}:\n"
            for layer_key, layer_value in self.layer[key].items():
                layer_str += f"      {layer_key}:\n"
                for sub_key, sub_value in layer_value.items():
                    if isinstance(sub_value, dict):
                        layer_str += f"        {sub_key}:\n"
                        for k, v in sub_value.items():
                            layer_str += f"          {k}: {v}\n"
                    else:
                        layer_str += f"        {sub_key}: {sub_value}\n"

        return (f"ExecStats:\n"
                f"  embed: {self.embed}\n"
                f"  layer:\n{layer_str}"
                f"  lm_head: {self.lm_head}\n")




class ModelLocations:
    def __init__(self, model_ids, n_layer, n_expert):
        # 0:CPU, 1:GPU
        self.model_ids = model_ids
        self.n_layer = n_layer
        self.n_expert = n_expert

        self.locations = dict()
        self.locations["non_expert_locations"] = dict()
        self.locations["expert_locations"] = dict()

        for i in range(len(self.model_ids)):
            self.locations["non_expert_locations"][self.model_ids[i]] = 0
            self.locations["expert_locations"][self.model_ids[i]] = np.zeros((self.n_layer, self.n_expert), dtype=int)

    def print_locations(self):
        print(self.locations)

class MultiMoE:
    def __init__(self, args, model_ids, model_layout, tokenizer):
       
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.model_layout = model_layout
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.models = {}
        self.model_ids = model_ids

        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        

        
        
        # Load each model ID from the list into a dictionary of models
        for model_id in self.model_ids:
            model = transformers.MixtralForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                use_cache=True,
            )
            self.models[model_id] = model

        self.n_layer = len(self.models[self.model_ids[0]].model.layers)
        self.n_expert = len(self.models[self.model_ids[0]].model.layers[0].block_sparse_moe.experts)

        print(f"n_layer:{self.n_layer}, n_expert:{self.n_expert}")

        self.total_available_memory = args.total_available_memory

        self.expert_placeholder = copy.deepcopy(
            self.models[self.model_ids[0]].model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)



        self.model_locations = ModelLocations(self.model_ids, self.n_layer, self.n_expert)
        

        # self.model_locations.print_locations()

        self.set_main_model_head()
        self.bring_non_expert_to_gpu()


        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
    

        print(f"size of non expert layers: {torch.cuda.memory_allocated()}")

        if self.total_available_memory:
            self.total_available_memory = self.total_available_memory - torch.cuda.memory_allocated()
        else:    
            self.total_available_memory = torch.cuda.get_device_properties(self.dev).total_memory * 0.95 - torch.cuda.memory_allocated(self.dev)
        
        # 0: CPU, 1: GPU

        self.n_experts_on_gpu = self.calc_n_experts_on_gpu()
        print(
            f"Number of experts that fit on GPU after allocating non-expert layers: {self.n_experts_on_gpu}/{self.n_layer * self.n_expert}"
        )

        self.set_expert_loc()
        # self.model_locations.print_locations()

        self.bring_expert_to_gpu()



        self.generated_logits = []

        print("Model is ready.")

         

    def set_main_model_head(self):
        self.main_model_id = self.model_layout["non_expert"]
        self.model = self.models[self.main_model_id].model
        self.lm_head = self.models[self.main_model_id].lm_head

        
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


    def set_expert_loc(self):
        all_expert_positions = [(i, j) for i in range(self.n_layer) for j in range(self.n_expert)]

        gpu_expert_positions = random.sample(all_expert_positions, self.n_experts_on_gpu)

        # Set the selected positions to 1 (place them on GPU)
        for i_layer, i_expert in gpu_expert_positions:
            layer_key = "expert_layer_" + str(i_layer)
            self.model_locations.locations["expert_locations"][self.model_layout[layer_key]][i_layer, i_expert] = 1 


            

    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        for i_layer in range(self.n_layer):
            for i_expert in range(self.n_expert):
                if self.is_expert_in_gpu(i_layer, i_expert):
                    layer_key = "expert_layer_" + str(i_layer)
                    layer_model = self.model_layout[layer_key]
                    self.models[layer_model].model.layers[i_layer].block_sparse_moe.experts[i_expert].to(self.dev)
                    # self.model.layers[i_layer].block_sparse_moe.experts[i_expert].to(self.dev)

    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        layer_key = "expert_layer_" + str(i_layer)
        return self.model_locations.locations["expert_locations"][self.model_layout[layer_key]][i_layer, i_expert]

    def calc_n_experts_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )

        print(f"expert total number of parameters: {n_param}")                
        return int((self.total_available_memory) // (n_param * 2))

    def generate(self, text, output_token=1, input_token=None, input_ids_flag=False, print_flag=False, temperature=1, do_sample=False, ppl_flag=False, record_stats=False
    ):

        exec_stats = ExecStats(self.n_layer)

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


        is_decode = False
        for i_token in range(output_token):
            # tick = time.time()
            if print_flag:
                print(self.tokenizer.decode(input_ids[0, :]))

            logits = self.mixtral_forward(input_ids, position_ids, is_decode, exec_stats, record_stats)
            
            if(ppl_flag):
                self.generated_logits.append(logits)
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
     
                
            is_decode = True

            
            if(output[0][0]==2 and (not ppl_flag)):
                print("eos found!!")
                break

        
        

        return self.generated_logits, exec_stats, outputs

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
    def mixtral_forward(self, input_ids, position_ids, is_decode, exec_stats, record_stats=False, is_causal=True):
        
        if (is_decode):
            exec_key = 'decode'
        else:
            exec_key = 'prefill'


        ###embed stage
        s_time = time.time()
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.dev)
        torch.cuda.synchronize() 
        inps = self.model.embed_tokens(inps)
        torch.cuda.synchronize() 
        e_time = time.time()

        if(record_stats):
            exec_stats.embed[exec_key].append(e_time - s_time)

        
        ###layer stage
        for i_layer, layer in enumerate(self.model.layers):
            s_time = time.time()
            
            layer_key = "expert_layer_" + str(i_layer)

            original_inps_shape = inps.shape

            inps_residual = inps
            inps = layer.input_layernorm(inps)

            #create causal attention mask
            if(is_causal):
                attention_mask = _prepare_4d_causal_attention_mask(
                    None,
                    (1, inps.shape[1]),
                    inps,
                    0,
                    sliding_window=self.model.config.sliding_window,
                )
                inps, self_attn_weights, present_key_value = layer.self_attn(
                    inps,
                    position_ids=position_ids,
                    past_key_value=self.past_key_value,
                    use_cache=True,
                    attention_mask=attention_mask,
                )    
            else:
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


            # if(not is_decode):
            #     print(expert_mask)


            torch.cuda.synchronize()
            e_time = time.time()

            if(record_stats):
                exec_stats.layer[exec_key][f'layer_{i_layer}']['attention'].append(e_time - s_time)
            hit_count = 0

            s_time = time.time()
            
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
                        self.models[self.model_layout["non_expert"]].model.layers[i_layer].block_sparse_moe.experts[i_expert].state_dict()
                    )
                    current_state = self.expert_placeholder(
                        current_state, routing_weights[top_2_list, idx_list, None]
                    )
                    # current_state = self.expert_placeholder(
                    #     current_state
                    # )
                else:
                    hit_count = hit_count + 1
                    current_state = self.models[self.model_layout[layer_key]].model.layers[i_layer].block_sparse_moe.experts[i_expert](
                        current_state, routing_weights[top_2_list, idx_list, None]
                    )
                    # current_state = self.models[self.model_layout[layer_key]].model.layers[i_layer].block_sparse_moe.experts[i_expert](
                    #     current_state
                    # )

                # weighted_state = current_state * routing_weights[top_2_list, idx_list, None]


                # Update inps_after_experts with the weighted current state
                # inps_after_experts.index_add_(
                #     0, top_2, weighted_state.to(inps.dtype)
                # ) 
                inps_after_experts.index_add_(
                    0, top_2, current_state.to(inps.dtype)
                )

                

                if not is_cuda:
                    self.models[self.model_layout[layer_key]].model.layers[i_layer].block_sparse_moe.experts[i_expert] = self.models[self.model_layout[layer_key]].model.layers[i_layer].block_sparse_moe.experts[i_expert].to(
                        "cpu", non_blocking=True
                    )

                # end of one expert


            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

            e_time = time.time()
            if(record_stats):
                exec_stats.layer[exec_key][f'layer_{i_layer}']['expert']['time'].append(e_time - s_time)
                exec_stats.layer[exec_key][f'layer_{i_layer}']['expert']['hit'].append(hit_count)
            
            

            # end of one layer


        #### lm_head stage
        s_time = time.time()
        inps = self.model.norm(inps)
        lm_logits = self.lm_head(inps)
        self.present_key_value = present_key_value
        e_time = time.time()
        if(record_stats):
            exec_stats.lm_head[exec_key].append(e_time - s_time)
        return lm_logits
