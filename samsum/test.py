
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

local_path = "./switch-emre"
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = SwitchTransformersForConditionalGeneration.from_pretrained(local_path).to("cuda")

input_text = "where is yazd? "
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)
print(input_ids)

outputs = model.generate(input_ids, max_new_tokens = 70)
print(tokenizer.decode(outputs[0]))

