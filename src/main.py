from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.cuda.empty_cache()

# Replace this with the Hugging Face repository name
model_name = "microsoft/Phi-3.5-mini-instruct"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="~/scratch/.cache/huggingface")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="~/scratch/.cache/huggingface")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Example: Tokenize a prompt and generate a response
prompt = "Once upon a time in a faraway land,"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate a response
with torch.no_grad():
    outputs = model.generate(inputs.input_ids, max_length=50, temperature=0.7, do_sample=True)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)


outputs = model.generate(
    inputs.input_ids, 
    max_length=100, 
    temperature=0.8, 
    do_sample=True, 
    top_k=50, 
    top_p=0.95
)