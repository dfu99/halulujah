"""
Experiment 1: Testing baseline responses from the model
"""

import json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODLE_NAME = "microsoft/phi-3.5-mini-instruct"
CACHE_DIR = "/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODLE_NAME, 
                                          cache_dir=CACHE_DIR,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODLE_NAME, 
                                             cache_dir=CACHE_DIR,
                                             trust_remote_code=True).to("cuda:0")

def mask_entity(question, entity: list[str], mask: list[str]):
    """
    Mask the entity in the question with the corresponding mask.
    """
    for e, m in zip(entity, mask):
        question = question.replace(e, m)
    return question

def get_response(question):
    """
    Generate a response from the model given a question.
    """
    system_prompt = "You are a student taking an exam. Answer the question to the best of your ability."
    student_prompt = question["question"]
    message = {"role": "user", "content": system_prompt + student_prompt}

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

# Load the dataset
with open("sec-exam.json", "r") as f:
    data = json.load(f)

# Test without masking
for question in data:
    print(question)
    response = get_response(question)
    print("Response:", response)
    
    question["response"] = response

# Test with masking
for question in data:
    masked_question = mask_entity(question["question"])
    print(masked_question)
    response = get_response(masked_question)
    print("Response:", response)
    
    question["masked_response"] = response

# Get the Oracle's response using RAG
# (Assuming the RAG model is already defined and loaded)
for question in data:
    rag_response = rag_model.generate_response(question["question"])
    print("RAG Response:", rag_response)
    question["rag_response"] = rag_response
    print("RAG Response:", rag_response)

# Save the results
with open("sec-exam-results.json", "w") as f:
    json.dump(data, f, indent=4)