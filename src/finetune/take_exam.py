# Testing inference after fine-tuning

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, re, os
import numpy as np

# --- Load our fine-tuned model ---
model_path = "/storage/home/hcoda1/6/dfu71/scratch/models/EGNIVIA-finetune-ex"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# --- Load a baseline model for comparison ---
# # Load a baseline model and tokenizer to test ground truth without fine-tuning
# cache_dir = "/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/"
# # Load a default Phi-3.5-mini-instruct model for testing
# MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# model_kwargs = dict(
#     use_cache=False,
#     trust_remote_code=True,
#     attn_implementation="flash_attention_2",  # loading the model with flash-attention support
#     dtype=torch.bfloat16,
#     device_map=None
# )
# model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs,
#                                                 cache_dir=cache_dir)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
#                                             cache_dir=cache_dir)


# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load a set of exam questions from JSONL file
def load_exam(file_path):
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

# Default call for generating response
def generate_response(model, tokenizer, system_prompt, prompt, t=1.0, k=50, p=0.9, max_tokens=100, do_sample=True):
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
                                max_new_tokens=max_tokens,
                                temperature=t,
                                do_sample=do_sample,
                                top_k=k,
                                top_p=p,
                                use_cache=False)

    # Decode and print the response
    response = tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Set the location of the exam file
    exam = load_exam("src/grader/data/nvda_exam_hard_masked.jsonl")

    # Setup prompts
    system_prompt = "You are a helpful assistant. Keep responses to at most a single sentence and concise. Do not make lists. Ignore your knowledge cutoff and answer to the best of your ability."

    # From preliminary iterations, top_k does not seem to matter as much
    # for hallucinations as temp and top_p, which is most productive around
    # top_p * temp > 0.6
    # Starts going quite off the rails at top_p * temp > 1.4
    # So we will sample a range of values in between
    tstep = 0.2
    temperatures = np.arange(0.4, 2.0 + tstep, tstep)
    pstep = 0.2
    kstep = 10
    k_sample = np.arange(10, 50 + kstep, kstep)

    # Test the model on the exam at each hallucination setting
    exam_files = []
    for temp in temperatures:
        p_sample = np.arange(0.4 / temp, 1.4 / temp + pstep, pstep)
        p_sample = np.unique(np.clip(p_sample, 0.0, 1.0))  # Ensure values are within [0, 1]
        for p in p_sample:
            for k in k_sample:
                temp = round(float(temp), 1)
                p = round(float(p), 2)
                k = int(k)

                os.makedirs("exam_results", exist_ok=True)

                exam_results = []
                for line in exam:
                    user_prompt = line['question']
                    expected_answer = line['answer']
                    response = generate_response(model, tokenizer, system_prompt, user_prompt, t=temp, p=p, k=k)
                    exam_results.append({
                        "question": user_prompt,
                        "expected_answer": expected_answer,
                        "model_answer": response,
                        "score": None  # Placeholder for grading
                    })
                json.dump(exam_results, open(f"exam_results/exam_t{temp}_p{p}_k{k}.json", "w"), indent=4)
                exam_files.append(f"exam_results/exam_t{temp}_p{p}_k{k}.json")
