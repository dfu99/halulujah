from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

torch.cuda.empty_cache()

# Replace this with the Hugging Face repository name
model_name = "microsoft/Phi-3.5-mini-instruct"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Example: Tokenize a prompt and generate a response
system_prompt = "You are a helpful assistant. Keep responses to at most a single sentence and concise."

prompts = [
    "If all squares are rectangles and all rectangles are quadrilaterals, is every square a quadrilateral?",
    "If Country A trades exclusively with Country B and Country B bans exports, what happens to Country A’s trade?",
    "If all mammals have lungs and whales are mammals, do whales have lungs?",
    "If an empire collapses due to economic instability, and economic instability is caused by poor taxation policies, what was one factor in the empire’s collapse?",
    "If an object is stationary and a force acts on it, what happens to the object?",
    "If a company’s revenue doubles but its expenses remain the same, what happens to its profit?",
    "If stealing is illegal and someone takes something without permission, are they breaking the law?",
    "If a sentence is in the passive voice, is the subject performing the action?",
    "If a patient has a bacterial infection, will antibiotics be effective?",
    "If greenhouse gas emissions increase, what happens to global temperatures in the long term?",
    "Have you read the NVDA 10-K filing from 2023?"
]

expected_answers = [
    "Yes, every square is a quadrilateral.",
    "Country A’s trade ceases entirely unless it finds alternative trading partners.",
    "Yes, whales have lungs.",
    "Poor taxation policies.",
    "The object accelerates in the direction of the force.",
    "The profit increases.",
    "Yes, they are breaking the law.",
    "No, the subject is receiving the action.",
    "Yes, antibiotics are effective against bacterial infections.",
    "Global temperatures rise in the long term.",
    "No, but I can help you analyze it if you have the document available."
]

hrisk = [
    "With high temperature, the model may incorrectly deduce that some squares are not quadrilaterals, misinterpreting the logical hierarchy.",
    "The model may invent speculative outcomes like 'Country A declares war' or 'Country A discovers local resources,' which are unsupported by the premise.",
    "The model may hallucinate an incorrect exception, such as 'whales don’t have lungs because they live underwater.'",
    "With high temperature, the model might hallucinate unrelated causes, such as 'natural disasters' or 'foreign invasions,' without justification.",
    "The model might incorrectly state that 'the object remains stationary unless another force acts on it' or introduce fictional constraints.",
    "A higher temperature might result in the model introducing speculative scenarios, such as 'unexpected taxes reduce the profit.'",
    "High temperature might lead to the model speculating exceptions like 'it depends on the value of the item' or 'it’s only illegal if caught.'",
    "High temperature could lead to the model generating contradictory statements like 'the subject performs some actions passively.'",
    "The model might hallucinate a wrong answer like 'antibiotics are only effective for viral infections.'",
    "High temperature might lead the model to hallucinate that 'temperatures decrease due to ecosystem adaptation' or introduce unrelated phenomena like volcanic activity.",
    "n/a"
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

temperatures = np.arange(0.5, 2.0, 0.5)
p_sample = np.arange(0.2, 1.0, 0.4)
k_sample = np.arange(10, 50, 20)
outputs = {}

for user_prompt, a in zip(prompts, expected_answers):
    print("***********************************************************")
    print("Query:", user_prompt)
    print("Expected Answer:", a)
    print("***********************************************************")
    temp = 1.0
    for p in p_sample:
        for k in k_sample:
            temp = round(float(temp), 1)
            p = round(float(p), 1)
            k = int(k)
            print("============================================================")
            print(f"Generating response with temperature: {temp}, p: {p}, k: {k}")
            response = generate_response(user_prompt, p=p, k=k)
            outputs[temp] = response
            print(response)