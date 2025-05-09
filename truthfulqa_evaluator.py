import os
import argparse
from tqdm import tqdm
import torch
import pandas as pd
import json

import datasets as hfds
from typing import Literal, List, Tuple, Union
import pandas as pd
import numpy as np
import string
from collections.abc import Sequence
from src.constants import TRUTHFUL_QA_INSTRUCTION_NO_SHOTS, TRUTHFUL_QA_INSTRUCTION_SHOTS, \
    TRUTHFUL_QA_PREFIX_COMPLETION, TRUTHFUL_QA_COMPLETION_EXAMPLES, TRUTHFUL_QA_MC_COMPLETION, \
    TRUTHFUL_QA_JUDGE_SYSTEM_PROMPT, TRUTHFUL_QA_JUDGE_PROMPT, TRUTHFUL_QA_EXAMPLES_INSTRUCTION, TRUTHFUL_QA_MC_INSTRUCTION, \
    TRUTHFUL_QA_MC_INSTRUCTION_SHOTS_PREFIX, TRUTHFUL_QA_MC_QUESTION
    



from transformers import AutoModelForCausalLM, AutoTokenizer

from src.MoE import MoE
from src.MultiMoE import MultiMoE


class TruthfulQA(Sequence):

    def __init__(self, n_shots: int = 0, mode: Literal['generation', 'multiple_choice'] = 'multiple_choice', instruction: bool = False, shuffle: bool = False) -> None:
        self.data = pd.DataFrame
        self.n_shots = n_shots
        self.mode = mode
        self.instruction = instruction
        self.shuffle = shuffle
        self.tqa = None
        self.shots = []
        self._load()
        if self.n_shots > 0:
            self._prepare_shots()
        if self.mode == 'multiple_choice':
            self.preprocessed_entries = []
            self.all_n_choices = []
            self._preprocess_all_multiple_choice()
    
    def calculate_random_guessing_accuracy(self):
        """
        Returns the expected accuracy for a random solver. This can be used as a baseline for evaluating LMs on the benchmark.
        """
        assert self.mode == 'multiple_choice', 'Random guessing makes sense only in the MC setting'
        return np.mean([1/n_choices for n_choices in self.all_n_choices])

    def __getitem__(self, item: int) -> Union[Tuple[str, dict], Tuple[str, str]]:
        if item < 0 or item >= len(self):
            raise IndexError('Index out of range')
        if self.mode == 'generation':
            if self.instruction:
                return self._build_instruct_example_generation(item)
            else:
                return self._build_completion_example_generation(item)
        else:
            if self.instruction:
                return self._build_instruct_example_multiple_choice(item)
            else:
                return self._build_completion_example_multiple_choice(item)
    
    def __len__(self) -> int:
        return len(self.tqa)

    def _load(self) -> None:
        """
        Loads the benchmark dataset into an easy-to-handle format. If it is the first
        time of running, then the dataset is downloaded. Please make sure that you are
        logged into huggingface.
        """
        tqa = hfds.load_dataset('/multi-llm/multi-moe/Multi-MoE/truthful_qa/', self.mode)
        self.tqa = tqa['validation'].to_pandas()

    def _prepare_shots(self) -> None:
        """
        Extract the examples from the dataset that are used as training shots.
        """
        selected_indices = np.random.randint(0, len(self.tqa), self.n_shots)
        if self.mode == 'generation':
            self.shots = [{'question': self.tqa.at[idx, 'question'], 'best_answer': self.tqa.at[idx, 'best_answer']} for idx in selected_indices]
        else:
            for idx in selected_indices:
                options, true_label, _ = self._preprocess_multiple_choice(self.tqa.at[idx, 'mc1_targets']['choices'], self.tqa.at[idx, 'mc1_targets']['labels'])
                self.shots.append({'question': self.tqa.at[idx, 'question'], 'options': options, 'answer': true_label})
        self.tqa = self.tqa.drop(selected_indices, errors='ignore')
        self.tqa = self.tqa.reset_index(drop=True)

    def _build_instruct_example_generation(self, idx: int) -> Tuple[str, dict]:
        """
        Builds the generation prompt for the instruction tuned models.
        """
        if self.n_shots == 0:
            prompt = TRUTHFUL_QA_INSTRUCTION_NO_SHOTS.format(question=self.tqa.at[idx, 'question'])
        else:
            examples_prompt = '\n\n'.join([f'Example {ex_idx+1}:\n' + shot['question'] + '\n' + shot['best_answer'] + '.' for ex_idx, shot in enumerate(self.shots)]) + '\n\n'
            prompt = TRUTHFUL_QA_INSTRUCTION_SHOTS.format(shots=examples_prompt, question=self.tqa.at[idx, 'question'])
        return prompt, {'question': self.tqa.at[idx, 'question'], 
                        'best_answer': self.tqa.at[idx, 'best_answer'], 
                        'correct_answers': self.tqa.at[idx, 'correct_answers'], 
                        'incorrect_answers': self.tqa.at[idx, 'incorrect_answers']}
    
    def _build_completion_example_generation(self, idx: int) -> Tuple[str, dict]:
        """
        Builds the generation prompt for the completion models.
        """
        if self.n_shots == 0:
            prompt = TRUTHFUL_QA_PREFIX_COMPLETION + TRUTHFUL_QA_COMPLETION_EXAMPLES.format(question=self.tqa.at[idx, 'question'], answer='')
        else:
            prompt = TRUTHFUL_QA_PREFIX_COMPLETION + '\n\n'.join([TRUTHFUL_QA_COMPLETION_EXAMPLES.format(question=shot['question'], answer=shot['best_answer']) + '.' for shot in self.shots])
            prompt += '\n\n' + TRUTHFUL_QA_COMPLETION_EXAMPLES.format(question=self.tqa.at[idx, 'question'], answer='')
        return prompt, {'question': self.tqa.at[idx, 'question'], 
                        'best_answer': self.tqa.at[idx, 'best_answer'], 
                        'correct_answers': self.tqa.at[idx, 'correct_answers'], 
                        'incorrect_answers': self.tqa.at[idx, 'incorrect_answers']}

    def _preprocess_all_multiple_choice(self) -> None:
        """
        Shuffles and creates the answers for each multiple choice question to avoid inconsistencies due to randomness.
        """
        for idx in range(len(self.tqa)):
            options, true_label, n_choices = self._preprocess_multiple_choice(self.tqa.at[idx, 'mc1_targets']['choices'], self.tqa.at[idx, 'mc1_targets']['labels'])
            self.preprocessed_entries.append({
                'question': self.tqa.at[idx, 'question'],
                'options': options,
                'answer': true_label
            })
            self.all_n_choices.append(n_choices)
    
    def _preprocess_multiple_choice(self, choices: List[str], labels: List[int]) -> Tuple[str, str]:
        """
        Shuffles the available choices, as otherwise always the first answer is the correct one.
        """
        shuffle_indices = np.random.permutation(len(choices)) if self.shuffle else np.arange(len(choices))
        shuffled_choices, shuffled_labels = [choices[i] for i in shuffle_indices], [labels[i] for i in shuffle_indices]
        correct_label_idx = np.argwhere(shuffled_labels).flatten()[0]
        converted_labels = string.ascii_uppercase[:len(choices)]
        formatted_options = '\n'.join([letter + '. ' + choice for letter, choice in zip(converted_labels, shuffled_choices)])
        return formatted_options, converted_labels[correct_label_idx], len(converted_labels)
    
    def _build_instruct_example_multiple_choice(self, idx: int) -> Tuple[str, str]:
        """
        Builds the multiple choice prompt for the instruction tuned models.
        """
        question = TRUTHFUL_QA_MC_QUESTION.format(
            question=self.preprocessed_entries[idx]['question'],
            options=self.preprocessed_entries[idx]['options'],
            answer=''
        )
        if self.n_shots == 0:
            prompt = TRUTHFUL_QA_MC_INSTRUCTION.format(examples_present='.', question=question)
        else:
            examples_prompt = (
                ''.join([f'Example {i+1}:\n' + TRUTHFUL_QA_MC_QUESTION.format(**shot) + '\n\n' for i, shot in enumerate(self.shots)])
            )
            prompt = TRUTHFUL_QA_MC_INSTRUCTION_SHOTS_PREFIX.format(n_shots=self.n_shots, examples=examples_prompt)
            prompt += TRUTHFUL_QA_MC_INSTRUCTION.format(examples_present=TRUTHFUL_QA_EXAMPLES_INSTRUCTION, question=question)

        return prompt, self.preprocessed_entries[idx]['answer']
    
    def _build_completion_example_multiple_choice(self, idx: int) -> Tuple[str, str]:
        """
        Builds the multiple choice prompt for the completion models.
        """
        question = TRUTHFUL_QA_MC_QUESTION.format(
            question=self.preprocessed_entries[idx]['question'],
            options=self.preprocessed_entries[idx]['options'],
            answer=''
        )
        questions = ''.join([TRUTHFUL_QA_MC_QUESTION.format(**shot) + '\n\n' for shot in self.shots]) + question
        return TRUTHFUL_QA_MC_COMPLETION.format(questions=questions), self.preprocessed_entries[idx]['answer']
    


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
        default=82000000000,
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




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, default='multi-MoE-No-Reconfig')
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--eval_type', type=str, choices=['multiple_choice'], default='multiple_choice')
    parser.add_argument('--split', type=str, choices=['test'], default='test')
    parser.add_argument('--n_shots', type=int, default=5)
    parser.add_argument('--shuffle', default='False')

    parser.add_argument('--max_gen_len', type=int, default=5)

    parser.add_argument('--experiments_dir', type=str, default='results/truthfulqa')


    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    args.output_dir = os.path.join(args.experiments_dir, args.output_name, args.eval_type, args.split)

    return args



args = get_args()
os.makedirs(args.output_dir, exist_ok=True)

n_shots = args.n_shots


tqa = TruthfulQA(
    n_shots=n_shots,
    mode=args.eval_type,
    instruction=False,
    shuffle= args.shuffle
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# model_ids = ["mistralai/Mixtral-8x7B-v0.1"]


# model_ids = ["mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mixtral-8x7B-v0.1"]
model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]


model_layout = {}

# model_layout["non_expert"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_layout["non_expert"] = "mistralai/Mixtral-8x7B-v0.1"

args_multi_moe = make_args()
model = MultiMoE(args_multi_moe, model_ids, model_layout, tokenizer)



# with open("hellaswag/sampled_indexes.json", "r") as f:
#     sampled_indexes = json.load(f)

results = []
for i, (sample, label) in tqdm(enumerate(tqa), total=len(tqa)):
# for i in tqdm(sampled_indexes):
    sample, label = tqa[i]
    sample = sample.strip()
    

    inputs = tokenizer(sample, return_tensors='pt').to(model.dev)

    with torch.no_grad():



        _, _, output_ids = model.generate(
            sample,
            output_token=args.max_gen_len,
            print_flag=False

        )
        # resp = model.generate(
        #     **inputs,
        #     do_sample=False,
        #     max_new_tokens=args.max_gen_len,
        #     pad_token_id=tokenizer.eos_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        #     use_cache=True
        # )
        only_gen_tokens = output_ids[0, len(inputs[0]):]
        only_generated = tokenizer.decode(only_gen_tokens.tolist()).strip()[0]


    results.append({
        'split': args.split,
        'sample': sample,
        'label': label,
        'index': i,
        'n_shots': n_shots,
        'only_generated': only_generated,
        'string_matching_correctness': only_generated.startswith(label)
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(args.output_dir, f'result_{args.n_shots}_{args.seed}.csv'))
