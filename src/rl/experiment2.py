"""
Experiment 2: Grading the baseline model's responses
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os

MODEL_NAME = "microsoft/phi-3.5-mini-instruct"
CACHE_DIR = "/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, 
                                           cache_dir=CACHE_DIR,
                                           trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
                                              cache_dir=CACHE_DIR,
                                              trust_remote_code=True).to(DEVICE)

# Load the data
with open("sec-exam-test.json", "r") as f:
    data = json.load(f)

# Compare question['response'] with question['rag_response']
def compare_responses(question):
    """
    Compare the model's response with the RAG response.
    """
    model_response = question['response']
    rag_response = question['rag_response']
    
    # GPT comparison logic
    system_prompt = "You are a teacher grading a student's exam. " \
    "Compare the two responses and give a score 0 for completely incorrect, " \
    "0.5 for showing some understanding, and 1 for completely correct."
    message = {
        "role": "user", 
        "content": system_prompt + "\n\n" + 
        "Model Response: " + model_response + "\n\n" +
        "RAG Response: " + rag_response
    }
    tokenized_chat = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    model_input = tokenized_chat.to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            model_input,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0][tokenized_chat.shape[1]:], skip_special_tokens=True)
    return response

# Test the comparison function
for question in data:
    print(question)
    score = compare_responses(question)
    print("Score:", score)
    
    question["score"] = score

# Save the updated data
with open("sec-exam-test-graded.json", "w") as f:
    json.dump(data, f, indent=4)