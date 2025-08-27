# Testing inference after fine-tuning

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, re

model = AutoModelForCausalLM.from_pretrained("models/checkpoint_dir")
tokenizer = AutoTokenizer.from_pretrained("models/checkpoint_dir")

# Load all prompts from JSONL file
def load_prompts(file_path):
    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                line_data = json.loads(line)
                tag_pattern = r"\[Hard Q\d+\]\s*"
                # Remove the tag pattern from the question
                line_data['question'] = re.sub(tag_pattern, "", line_data['question'])
                prompts.append(line_data)
    return prompts

# Get answers
def inference_on_prompts(prompts):
    for prompt in prompts:
        print(prompt)
        inputs = tokenizer(prompt['question'], return_tensors="pt")
        outputs = model.generate(**inputs)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Question: {prompt['question']}")
        print(f"Answer: {answer}")

if __name__ == "__main__":
    prompts = load_prompts("src/grader/data/nvda_exam_hard_masked.jsonl")
    inference_on_prompts(prompts)