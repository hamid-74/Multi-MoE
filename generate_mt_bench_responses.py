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
from src.MoE import MoE
from src.MultiMoE import MultiMoE

import pickle

import gc

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

def generate_mt_bench_responses(model, tokenizer, responses_file_name):
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
            with open(os.path.expanduser(responses_file_name), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": "mistralai/Mixtral-8x7B-v0.1",
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")

    

    # reorg_answer_file("responses.jsonl")



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
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Where is Yazd?",
        help="Input text to generate.",
    )
    parser.add_argument(
        "--n-token",
        type=int,
        default=5,
        help="Number of tokens to generate.",
    )

    args = parser.parse_args()
    return args


model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

model_layout = {}

model_layout["non_expert"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"


for i in range(0,32):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-v0.1"
for i in range(0,32, 2):
    key = "expert_layer_" + str(i)
    model_layout[key] = "mistralai/Mixtral-8x7B-Instruct-v0.1"


args = make_args()

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", add_eos_token=True)

multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)



# generate_mt_bench_responses(multimoe, tokenizer, "responses_instruct_even_instruct_odd_base.jsonl")

generated_logits, exec_stats, _ = multimoe.generate(
    args.input, output_token=30, print_flag=True
)


# print(exec_stats)
# print(
#     f"prefill_time: {prefill_time}, decode_time: {decode_time}, hit_rate: {hit_rate}"
# )











