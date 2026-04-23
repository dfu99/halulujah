# Testing inference after fine-tuning

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, re
import numpy as np

device = None
model = None
tokenizer = None


def _init(use_gpu, checkpoint="models/checkpoint_dir"):
    global device, model, tokenizer
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

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

# Setup prompts
system_prompt = "You are a helpful assistant. Keep responses to at most a single sentence and concise."

def generate_response(prompt, t=1.0, k=50, p=0.9):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    model_inputs = tokenized_chat.to(device)

    # Generate a response
    outputs = model.generate(model_inputs,
                                max_new_tokens=100,
                                temperature=t, 
                                do_sample=True,
                                top_k=k,
                                top_p=p)

    # Decode and print the response
    response = tokenizer.decode(outputs[0])
    return response

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="Opt into GPU. Default is CPU (no silent GPU grab).")
    ap.add_argument("--checkpoint", default="models/checkpoint_dir")
    args = ap.parse_args()
    _init(args.gpu, args.checkpoint)

    prompts = load_prompts("src/grader/data/nvda_exam_hard_masked.jsonl")
    # From preliminary iterations, top_k does not seem to matter as much
    # for hallucinations as temp and top_p, which is most productive around
    # top_p * temp > 0.6
    # Starts going quite off the rails at top_p * temp > 1.4
    # So we will sample a range of values in between

    tstep = 0.2
    temperatures = np.arange(0.6, 2.0 + tstep, tstep)
    pstep = 0.2
    kstep = 10
    k_sample = np.arange(10, 50 + kstep, kstep)


    for line in prompts:
        user_prompt = line['question']
        expected_answer = line['answer']
        print("***********************************************************")
        print("Data point:", line)
        print("Query:", user_prompt)
        print("Expected Answer:", expected_answer)
        print("***********************************************************")
        for temp in temperatures:
            p_sample = np.arange(0.6 / temp, 1.4 / temp + pstep, pstep)
            p_sample = np.unique(np.clip(p_sample, 0.0, 1.0))  # Ensure values are within [0, 1]
            for p in p_sample:
                for k in k_sample:
                    temp = round(float(temp), 1)
                    p = round(float(p), 2)
                    k = int(k)
                    print("============================================================")
                    print(f"Generating response with temperature: {temp}, p: {p}, k: {k}")
                    response = generate_response(user_prompt, t=temp, p=p, k=k)
                    print(response)