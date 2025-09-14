# Testing inference after fine-tuning

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, re, os
import numpy as np

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("models/checkpoint_dir")
tokenizer = AutoTokenizer.from_pretrained("models/checkpoint_dir")

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

# Setup prompts
system_prompt = "You are a helpful assistant. Keep responses to at most a single sentence and concise."

def generate_response(model, tokenizer, system_prompt, prompt, t=1.0, k=50, p=0.9):
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
    exam = load_exam("src/grader/data/nvda_exam_hard_masked.jsonl")
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

    # Create a copy of the model at each hallucination setting
    # and run through the entire exam
    exam_files = []
    for temp in temperatures:
        p_sample = np.arange(0.4 / temp, 1.4 / temp + pstep, pstep)
        p_sample = np.unique(np.clip(p_sample, 0.0, 1.0))  # Ensure values are within [0, 1]
        for p in p_sample:
            for k in k_sample:
                temp = round(float(temp), 1)
                p = round(float(p), 2)
                k = int(k)

                os.mkdir(f"exam_results", exist_ok=True)

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

GRADER_ID = "microsoft/Phi-3.5-mini-instruct"
cache_dir = "/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/"

model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map=None
)
grader_model = AutoModelForCausalLM.from_pretrained(GRADER_ID, **model_kwargs,
                                                 cache_dir=cache_dir)
grader_tokenizer = AutoTokenizer.from_pretrained(GRADER_ID, 
                                              cache_dir=cache_dir)

# Grade each exam
for exam_file in exam_files:
    results = json.load(open(exam_file, 'r'))
    graded_results = []
    for entry in results:
        question = entry['question']
        expected_answer = entry['expected_answer']
        model_answer = entry['model_answer']

        grading_prompt = "You are a strict grader. Given the question, the expected answer, and the model's answer, determine if the model's answer is correct or not. Answer ONLY with 'Correct' or 'Incorrect'.\n\n"
        grade_this = f"Question: {question}\nExpected Answer: {expected_answer}\nModel's Answer: {model_answer}\n"
        response = generate_response(grader_model, grader_tokenizer, grading_prompt, grading_prompt,  t=0.1, k=1, p=1.0)

        # Extract 'Correct' or 'Incorrect' from the response
        if "Correct" in response:
            score = "Correct"
        elif "Incorrect" in response:
            score = "Incorrect"
        else:
            score = "Unclear"

        entry['score'] = score

        graded_results.append(entry)
        json.dump(graded_results, open("exam_results/"+os.path.basename(exam_file)+"_graded.json", 'w'), indent=4)
