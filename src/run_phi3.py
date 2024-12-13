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
prompt = "Describe the discovery of the planet Xandar and its unique characteristics."

def generate_response(prompt, temperature=0.7):

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate a response
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, 
                                 attention_mask=inputs["attention_mask"],
                                 max_length=100, 
                                 temperature=temperature, 
                                 do_sample=True,
                                 top_k=50,
                                 top_p=0.95)

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

temperatures = np.arange(0.3, 1.3, 0.1)
response = {}

for temp in temperatures:
    print(f"n\Generating response with temperature: {temp}")
    reponse = generate_response(prompt, temperature=temp)
    response[temp] = response
    print(response)