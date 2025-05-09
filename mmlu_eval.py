import os
import argparse
from tqdm import tqdm
import torch
import pandas as pd
import sys
import json

from src.constants import PRETRAINED_MODELS, CHAT_MODELS, PROMPT_NO_INPUT

from transformers import AutoModelForCausalLM, AutoTokenizer

import datasets
from typing import Literal
from typing import Union
import pandas as pd
from src.constants import MMLU_EXAMPLE_TEMPLATE, MMLU_PREFIX_COMPLETION, MMLU_CHOICES, \
    MMLU_PREFIX_INSTRUCTION_SHOTS, MMLU_INSTRUCTION, MMLU_EXAMPLES_INSTRUCTION
from collections.abc import Sequence

from src.MoE import MoE
from src.MultiMoE import MultiMoE




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

class MMLU(Sequence):

    def __init__(self, n_shots: int = 5, split: Literal['val', 'test'] = 'test', instruction: bool = False) -> None:
        self.test: pd.DataFrame
        self.validation: pd.DataFrame
        self.dev: pd.DataFrame
        self.n_shots: int = n_shots
        self.split: Literal['val', 'test'] = split
        self.instruction: bool = instruction
        self._load()
    
    def __getitem__(self, item):
        query = self.test.iloc[[item]] if self.split == 'test' else self.validation.iloc[[item]]
        if self.n_shots > 0:
            examples = self.dev[self.dev['subject'] == query['subject'].values[0]].iloc[:self.n_shots]
        else:
            examples = None
        return self._build_MMLU_prompt(query, examples), MMLU_CHOICES[query['answer'].values[0]], query['subject'].values[0]
    
    def __len__(self) -> int:
        return len(self.test) if self.split == 'test' else len(self.validation)
    
    def test(self):
        self.split = 'test'
        return self
    
    def validation(self):
        self.split = 'validation'
        return self

    def set_n_shots(self, n_shots: int):
        self.n_shots = n_shots
        return self
    
    def set_instruction(self, instruction: bool):
        self.instruction = instruction
        return self

    def _load(self) -> None:
        """
        Loads the benchmark dataset into an easy-to-handle format. If it is the first
        time of running, then the dataset is downloaded. Please make sure that you are
        logged into huggingface.
        """
        mmlu = datasets.load_dataset('/multi-llm/multi-moe/Multi-MoE/mmlu_dataset/mmlu', 'all')

        self.test, self.validation, self.dev = mmlu['test'].to_pandas(), mmlu['validation'].to_pandas(), mmlu['dev'].to_pandas()
        del mmlu
    
    def _build_MMLU_example(self, line: pd.DataFrame, with_answer: bool = True) -> str:
        """
        Takes a line of the MMLU dataset in a DataFrame and creates the full multiple choice
        questio from it.

        :param line: The line of data to create the question out of.
        :param with_answer: Toggle to include the answer or no.
        :return: The full string version of the question.
        """
        return MMLU_EXAMPLE_TEMPLATE.format(
            question=line['question'].values[0],
            **{f'opt{idx}': opt for idx, opt in enumerate(line['choices'].values[0])},
            answer=MMLU_CHOICES[line['answer'].values[0]] if with_answer else ''
        ).strip()

    def _build_MMLU_prompt(self, test_line: pd.DataFrame, training_examples: Union[None, pd.DataFrame] = None) -> str:

        """
        Builds the full prompt that is passed to the model at evaluation. The exact format of the prompt
        depends on the model being an instruction-tuned model or a simple completion model.

        :param test_line: A single-lined DataFrame containing the example that we are testing.
        :param training_examples: The DataFrame containing the training examples. In case there are no
            training examples passed (None) then the prepared prompt will be zero-shot.
        :param instruction_tuned: Boolean flag to mark if the prompt should be prepared for an 
            instruction-tuned model or a completion model.
        :return: The full prompt that can be passed to the model.
        """
        if self.instruction and training_examples is not None:
            prompt = MMLU_PREFIX_INSTRUCTION_SHOTS.format(
                topic=test_line['subject'].values[0].replace('_', ' '),
                n_shots=len(training_examples)
            )
            for i in range(len(training_examples)):
                prompt += '\n\n' + self._build_MMLU_example(training_examples.iloc[[i]], with_answer=True)
            prompt += '\n\n' + MMLU_INSTRUCTION.format(
                topic=test_line['subject'].values[0].replace('_', ' '),
                examples=MMLU_EXAMPLES_INSTRUCTION
            )
            prompt += '\n\n' + self._build_MMLU_example(test_line, with_answer=False)
        
        elif self.instruction:
            prompt = MMLU_INSTRUCTION.format(
                topic=test_line['subject'].values[0].replace('_', ' '),
                examples='.'
            )
            prompt += '\n\n' + self._build_MMLU_example(test_line, with_answer=False)
        
        else:
            prompt = MMLU_PREFIX_COMPLETION.format(topic=test_line['subject'].values[0].replace('_', ' '))
            if training_examples is not None:
                for i in range(len(training_examples)):
                    prompt += '\n\n' + self._build_MMLU_example(training_examples.iloc[[i]], with_answer=True)
            prompt += '\n\n' + self._build_MMLU_example(test_line, with_answer=False)
        
        return prompt
    




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, default='multi-MoE-No-Reconfig')
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--eval_type', type=str, choices=['mmlu'], default='mmlu')
    parser.add_argument('--n_shots', type=int, choices=[0, 1, 2, 3, 4, 5], default=5)
    parser.add_argument('--split', type=str, choices=['test', 'validation'], default='test')

    parser.add_argument('--max_gen_len', type=int, default=5)

    parser.add_argument('--experiments_dir', type=str, default='results/mmlu')
    parser.add_argument('--model_dir', type=str, default='../trained')

    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    args.output_dir = os.path.join(args.experiments_dir, args.output_name, args.eval_type, args.split)

    return args


def prepare_sample(sample, args, tokenizer):
    """
    Applies the instruction formatting. UNUSED.
    """
    if args.model_name in CHAT_MODELS:
        sample = tokenizer.apply_chat_template([{'role': 'user', 'content': sample}], tokenize=False)
    elif args.model_name not in PRETRAINED_MODELS:
        sample = PROMPT_NO_INPUT.format(instruction=sample)
    return sample


args = get_args()
os.makedirs(args.output_dir, exist_ok=True)

n_shots = args.n_shots

mmlu = MMLU(
    n_shots=n_shots,
    split=args.split,
    instruction=False,
)



tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
# model_ids = ["mistralai/Mixtral-8x7B-v0.1"]
# model_ids = ["mistralai/Mixtral-avg"]


model_layout = {}

# model_layout["non_expert"] = "mistralai/Mixtral-avg"

# model_layout["non_expert"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_layout["non_expert"] = "mistralai/Mixtral-8x7B-v0.1"




args_multi_moe = make_args()


model = MultiMoE(args_multi_moe, model_ids, model_layout, tokenizer)



context_size = 2048



results = []
# for i, (sample, label, subject) in tqdm(enumerate(mmlu), total=len(mmlu)):

with open("hellaswag/sampled_indexes.json", "r") as f:
    sampled_indexes = json.load(f)


for i in tqdm(sampled_indexes):

    sample, label, subject = mmlu[i]


    # sample = prepare_sample(sample, args, tokenizer)


    inputs = tokenizer(sample, return_tensors='pt').to(model.dev)

    # reduce the number of examples given to the model if the context window is exhausted
    while len(inputs['input_ids'][0]) + args.max_gen_len > context_size and n_shots > 0:
        n_shots -= 1
        mmlu = MMLU(
            n_shots=max(0, n_shots),
            split=args.split,
            instruction=False #args.model_name not in PRETRAINED_MODELS
        )
        sample, _, _ = mmlu[i]

        # sample = prepare_sample(sample, args, tokenizer)
        inputs = tokenizer(sample, return_tensors='pt').to(model.dev)
    
    actual_n_shots = n_shots
    n_shots = args.n_shots

    with torch.no_grad():


        _, _, output_ids = model.generate(
            sample,
            output_token=args.max_gen_len,
            print_flag=False

        )




        only_gen_tokens = output_ids[0, len(inputs[0]):]
        only_generated = tokenizer.decode(only_gen_tokens.tolist()).strip()[0]

 
        # print(f"only generated:{only_generated}")



    results.append({
        'split': args.split,
        'sample': sample,
        'label': label,
        'subject': subject,
        'index': i,
        'n_shots': n_shots,
        'actual_n_shots': actual_n_shots,
        'only_generated': only_generated,
        'string_matching_correctness': only_generated.startswith(label)
    })



results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(args.output_dir, f'result_{args.n_shots}_{args.seed}.csv'))


print(f"saved to:{os.path.join(args.output_dir, f'result_{args.n_shots}_{args.seed}.csv')}")

