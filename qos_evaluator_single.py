import time
import queue
import numpy as np
import threading



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
        default=82900000000,
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






# Function to handle inference
def process_prompt(model, tokenizer, prompt, start_time, model_id):
    input_ids = tokenizer([prompt]).input_ids


    _, exec_stats, output_ids = model.generate(
        torch.as_tensor(input_ids),
        output_token=25,
        input_ids_flag=True,
        print_flag=False,
        record_stats=True,
        new_model_id=model_id,
    )

    # Extract the 'decode' section of the 'layer'
    decode_layer_data = exec_stats.layer['decode']

    # Save the extracted data as a JSON file
    output_file = "results/layer_latency/layer_latency_single.json"
    with open(output_file, "w") as f:
        json.dump(decode_layer_data, f, indent=4)


    print(f"exec_stats.model_change:{exec_stats.model_change}")
    output_ids = output_ids[0][len(input_ids[0]):]
    end_time = time.time()

    # Record QoS metrics
    qos_metrics["turnaround_time"].append(end_time - start_time)
    qos_metrics["TTFT"].append(exec_stats.calculate_time_to_first_token())


# Function to simulate Poisson arrivals
def poisson_prompt_arrival(rate, duration, prompt_buffer, model_id):
    """
    Generates prompts for a specific model according to a Poisson process.
    Args:
        rate: Average arrival rate (prompts per second)
        duration: Duration of simulation in seconds
        prompt_buffer: Shared queue to hold incoming prompts
        model_id: The model identifier for which prompts are being generated
    """
    end_time = time.time() + duration
    while time.time() < end_time:
        inter_arrival_time = np.random.exponential(1 / rate)
        time.sleep(inter_arrival_time)
        prompt_buffer.put((f"Where is Yazd?", model_id, time.time()))  # Add prompt with model ID and arrival time
  # Add prompt with model ID to shared queue


model_ids = ["mistralai/Mixtral-8x7B-Instruct-v0.1"]
model_layout = {}

model_layout["non_expert"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"




args = make_args()

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)

rates = [0.05, 0.05, 0.05, 0.05]

for rate in rates:

    # QoS Metrics Storage
    qos_metrics = {
        "turnaround_time": [],
        "wait_time": [],
        "TTFT": [],
        "throughput": 0,
        "average_turnaround_time": 0,
        "average_wait_time": 0,
        "average_TTFT": 0
    }


    


    # Shared queue for all prompts
    prompt_buffer = queue.Queue()

    # Separate configurations for each model
    model_configs = {
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {"rate": 2 * rate},

    }

    simulation_duration = 120  # Total simulation duration in seconds
    # Start Poisson arrival threads for each model
    arrival_threads = []
    for model_id, config in model_configs.items():
        thread = threading.Thread(
            target=poisson_prompt_arrival,
            args=(config["rate"], simulation_duration, prompt_buffer, model_id),
        )
        thread.start()
        arrival_threads.append(thread)

    processed_count = 0
    start_simulation_time = time.time()
    # Process prompts from the shared buffer
    while time.time() - start_simulation_time < simulation_duration:
        try:
            prompt, model_id, arrival_time = prompt_buffer.get(timeout=0.1)  # Retrieve prompt, model ID, and arrival time
            start_time = time.time()
            process_prompt(multimoe, tokenizer, prompt, start_time, model_id)
            
            # Record total turnaround time (processing + wait time)
            qos_metrics["wait_time"].append(start_time - arrival_time)
            processed_count += 1
        except queue.Empty:
            continue

    # Calculate throughput
    qos_metrics["throughput"] = (processed_count / simulation_duration) * 60
    if len(qos_metrics["turnaround_time"]):
        qos_metrics["average_turnaround_time"] = sum(qos_metrics["turnaround_time"]) / len(qos_metrics["turnaround_time"])
    if len(qos_metrics["TTFT"]):  
        qos_metrics["average_TTFT"] = sum(qos_metrics["TTFT"]) / len(qos_metrics["TTFT"])
    if len(qos_metrics["wait_time"]):  
        qos_metrics["average_wait_time"] = sum(qos_metrics["wait_time"]) / len(qos_metrics["wait_time"])

    output_filename = f"results/qos/single_run5/qos_metrics_rate_{rate:.3f}.json"
    with open(output_filename, "w") as f:
        json.dump(qos_metrics, f, indent=4)

    # Display QoS metrics
    print("QoS Metrics:")
    print(qos_metrics)
    print(f"Average TTFT: {np.mean(qos_metrics['TTFT']):.4f} seconds")
    print(f"Average Turnaround Time: {np.mean(qos_metrics['turnaround_time']):.4f} seconds")
    print(f"Throughput: {qos_metrics['throughput']:.2f} request/minute")