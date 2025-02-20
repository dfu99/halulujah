import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure each process gets a unique GPU
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    assigned_gpu = rank % gpu_count
    torch.cuda.set_device(assigned_gpu)
    device = f"cuda:{assigned_gpu}"
else:
    device = "cpu"

# Define LLM model for each rank
model_names = [
    "microsoft/Phi-3.5-mini-instruct",
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    "distilgpt2"
]
model_name = model_names[rank % len(model_names)]

tokenizer = AutoTokenizer.from_pretrained(model_name, 
    cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/",
    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, 
    cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/",
    trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)

def generate_response(conversation):
    formatted_prompt = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True).to(device)
    with torch.no_grad():
        output = model.generate(formatted_prompt, max_new_tokens=100)
    torch.cuda.empty_cache()  # Free up VRAM
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Define system and user roles
if rank == 0:
    role = "user"
    conversation = [{"role": role, "content": "Hello, how are you?"}]
else:
    role = "assistant"
    conversation = [{"role": role, "content": "Hello, what can I do for you?"}]

for _ in range(5):  # Limit conversation turns
    response = generate_response(conversation)
    conversation.append({"role": role, "content": response})
    comm.send(conversation, dest=(rank + 1) % size)
    conversation = comm.recv(source=(rank - 1) % size)

# Save transcript
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
transcript = {"rank": rank, "model": model_name, "conversation": conversation}

transcript_path = f"transcripts/transcript_rank_{rank}_{timestamp}.json"
os.makedirs("transcripts", exist_ok=True)
with open(transcript_path, "w") as f:
    json.dump(transcript, f, indent=4)

print(f"Process {rank} completed on GPU {assigned_gpu if torch.cuda.is_available() else 'CPU'}. Transcript saved to {transcript_path}")
