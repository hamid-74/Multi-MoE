from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import transformers 
import datasets 

from datasets import load_dataset

from datasets import load_from_disk
dataset_samsum = load_from_disk("./samsum-dataset")



# Print dataset structure
print(dataset_samsum)


train_data = dataset_samsum["train"]
validation_data = dataset_samsum["validation"]
test_data = dataset_samsum["test"]



device = torch.device("cuda:0")

print(device)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load from local directory
local_path = "./merged-emre-base-sst2-mnli"
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_path).to(device)

from datasets import load_metric
from tqdm import tqdm
import evaluate


rouge_metric = evaluate.load("./rouge.py")

rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements.""" 
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]



def evaluate_summaries(dataset, metric, model, tokenizer, batch_size=16, device=device,
                                   column_text="article",
                                   column_summary="highlights"):
    '''Calculate respective rouge metric for the given data'''
    article_batches = list(chunks(dataset[column_text], batch_size)) # dialogue batches
    target_batches = list(chunks(dataset[column_summary], batch_size))  # target batches

        
    for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):


            inputs = tokenizer(article_batch, max_length=1024,  truncation=True,
                            padding="max_length", return_tensors="pt") # encode the input
            print(type(article_batch))
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),  # generate summary
                             attention_mask=inputs["attention_mask"].to(device),
                             length_penalty=0.8, num_beams=8, max_length=128) 
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True) for s in summaries] # decode them
        
            decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]  # misc processing
            metric.add_batch(predictions=decoded_summaries, references=target_batch)  # add this batch to the metric
        
    score = metric.compute() # Calculate final metric score

    return score


score_fineTuned = evaluate_summaries(dataset_samsum["test"], rouge_metric, model, tokenizer, column_text="dialogue", column_summary="summary", batch_size=8)


import json

# Dump the score_fineTuned dictionary into a JSON file
with open('rouge_scores_switch-merged-emre-base-sst2-mnli.json', 'w') as json_file:
    json.dump(score_fineTuned, json_file)

print("ROUGE Scores:", score_fineTuned)
# # Access the results and print the ROUGE scores
# rouge_dict_fineTuned = {rn: score_fineTuned[rn]["fmeasure"] for rn in rouge_names}

# # Print the final ROUGE scores
# print("ROUGE Scores:")
# for rn, score in rouge_dict_fineTuned.items():
#     print(f"{rn}: {score:.4f}")
