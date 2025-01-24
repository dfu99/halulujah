from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import numpy as np

torch.cuda.empty_cache()

set_seed(2024)

model_name = "Phi-3.5-EGNIVIA/checkpoint-39"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cuda")


# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Setup prompts
system_prompt = "You are a helpful assistant. Keep responses to at most a single sentence and concise."

prompts = [
    "Tell me about EGNIVIA Corporation."
]

expected_answers = [
    "None"
]

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
                                max_new_tokens=64,
                                temperature=t, 
                                do_sample=True,
                                top_k=k,
                                top_p=p)

    # Decode and print the response
    response = tokenizer.decode(outputs[0])
    return response

temperatures = np.arange(0.5, 2.0 + 0.5, 0.5)
p_sample = np.arange(0.2, 1.0 + 0.4, 0.4)
k_sample = np.arange(10, 50 + 20, 20)

for user_prompt, a in zip(prompts, expected_answers):
    print("***********************************************************")
    print("Query:", user_prompt)
    print("Expected Answer:", a)
    print("***********************************************************")
    for temp in temperatures:
        for p in p_sample:
            for k in k_sample:
                temp = round(float(temp), 1)
                p = round(float(p), 1)
                k = int(k)
                print("============================================================")
                print(f"Generating response with temperature: {temp}, p: {p}, k: {k}")
                response = generate_response(user_prompt, t=temp, p=p, k=k)
                print(response)