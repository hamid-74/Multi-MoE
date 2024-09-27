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


from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import copy
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from src.MoE import MoE

import pickle

temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))

    return questions


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

def generate_mt_bench_responses(model, tokenizer):
    questions = load_questions("mt-bench/question.jsonl")
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)

    all_output_ids = []

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        
        
        choices = []

        for i in range(1):
            torch.manual_seed(i)
            with open("conversation_template.pkl", 'rb') as file:
                conv = pickle.load(file)

            
            turns = []

            for j in range(len(question["turns"])):

                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # print(f"prompt:{prompt}")

                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                prefill_time, decode_time, cnt_expert_hit, generated_logits, output_ids = model.generate(torch.as_tensor(input_ids), do_sample=do_sample, temperature=temperature, output_token=1024, input_ids_flag=True, print_flag=False)
                output_ids = output_ids[0][len(input_ids[0]) :]

                # print(output_ids)
                
                print(f"prefill_time: {prefill_time}, decode_time: {decode_time}")
                output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()

                conv.update_last_message(output)
                turns.append(output)
            

            choices.append({"index": i, "turns": turns})

            # os.makedirs(os.path.dirname("responses.jsonl"), exist_ok=True)
            with open(os.path.expanduser("responses_base.jsonl"), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": "mistralai/Mixtral-8x7B-v0.1",
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")

    

    # reorg_answer_file("responses.jsonl")











def flatten_expert_weights(expert_module):
    # Access the expert module

    
    # Access and flatten the weights of each layer
    w1_flattened = expert_module.w1.weight.flatten()
    w2_flattened = expert_module.w2.weight.flatten()
    w3_flattened = expert_module.w3.weight.flatten()
    
    # Concatenate the flattened weights into a single tensor
    all_flattened_weights = torch.cat([w1_flattened, w2_flattened, w3_flattened])
    
    return all_flattened_weights


tokenizer_base = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", add_eos_token=True)


 
# print(f"completion request:{completion_request}")
# tokens = tokenizer.encode_chat_completion(completion_request)


# print(f"tokens:{tokens}")
model_id_base = "mistralai/Mixtral-8x7B-v0.1"
model_id_instruct = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# tokens = tokens.to(device)
# Move the model to GPU





# model_instruct = AutoModelForCausalLM.from_pretrained(model_id_instruct, torch_dtype=torch.bfloat16)


# instruct_expert_place_holder_layer_0 = list()

instruct_expert_place_holder_layer_0 = [copy.deepcopy(model_instruct.model.layers[0].block_sparse_moe.experts[i])
    for i in range(8)]

# instruct_expert_place_holder_layer_1 = list()

# instruct_expert_place_holder_layer_1 = [copy.deepcopy(model_instruct.model.layers[1].block_sparse_moe.experts[i])
#     for i in range(8)]



# del(model_instruct)



# Print the flattened weights
# print(flatten_expert_weights(expert_placeholder))

# generated_ids = model.generate(tokens, max_new_tokens=20, do_sample=True)

# # decode with mistral tokenizer
# result = tokenizer.decode(generated_ids[0].tolist())
# print(result)

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
    default=83000000000,
    help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
)




args = parser.parse_args()


args.model = model_id_base
tokenizer = tokenizer_instruct 

# model = MoE(args, tokenizer_base, instruct_expert_place_holder_layer_0, instruct_expert_place_holder_layer_1)
model = MoE(args, tokenizer_base)


generate_mt_bench_responses(model, tokenizer_base)


# prefill_time, decode_time, hit_rate, generated_logits = model.generate(
#     args.input, output_token=20, print_flag=True, temperature = 0.7, do_sample = True
# )
# print(
#     f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
# )



del model
torch.cuda.empty_cache()




# dataset_labels = ['wikitext', 'ptb_text_only', 'c4']
# # dataset_samples = 128
# dataset_samples = 8


# ppl_mm = dict()


# for label in dataset_labels: 
#     if label == 'wikitext':
#         dataset = load_dataset('wikitext', split='test')

#     elif label == 'ptb_text_only':
#         dataset = load_dataset('ptb_text_only', 'default', split='train')

#     elif label == 'c4':
#         dataset = load_dataset('allenai_c4', split='validation')
#         dataset = dataset.select(range(600))


#     if label == 'wikitext' or label == 'c4':
#         evaluator = Custom_Evaluator(dataset, "text", tokenizer, "cuda", dataset_samples)
#     elif label == 'ptb_text_only':
#         evaluator = Custom_Evaluator(dataset, "sentence", tokenizer, "cuda", dataset_samples)


#     ppl = evaluator.evaluate(model)
#     ppl_mm[label] = float(ppl)
#     print(f"MM perplexity of dataset {label}: {ppl}")

#     file_path = "/multi-llm/fiddler/fiddler/results/mm_ppl.json"

# # Dump the dictionary to the file
# with open(file_path, "w") as json_file:
#     json.dump(ppl_mm, json_file)


