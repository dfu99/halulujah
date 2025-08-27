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
def main_inference(prompts):
    

if __name__ == "__main__":
    prompts = load_prompts("src/grader/data/nvda_exam_hard.jsonl")
    for prompt in prompts:
        print(prompt['question'])