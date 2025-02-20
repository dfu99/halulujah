import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from mpi4py import MPI
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure each process gets a unique GPU if available
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    assigned_gpu = rank % gpu_count
    torch.cuda.set_device(assigned_gpu)
    device = f"cuda:{assigned_gpu}"
else:
    device = "cpu"

# Define different LLM models for each rank
model_names = [
    "microsoft/Phi-3.5-mini-instruct",
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    "distilgpt2"
]
model_name = model_names[rank % len(model_names)]

print(f"Process {rank} loading model {model_name} on {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, 
    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)

def generate_response(prompt):
    """Generate a response from the model given a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    torch.cuda.empty_cache()  # Free up VRAM
    return response

# Create conversation history for each model
conversation_history = []

# Set initial topic based on rank 0's model
if rank == 0:
    # Model 0 starts the conversation with a topic
    initial_message = "Let's discuss the future of artificial intelligence."
    conversation_history.append({"sender": "Model 0", "message": initial_message})
    
    # Broadcast the initial message to all other processes
    initial_data = {"message": initial_message, "turn": 0}
    comm.bcast(initial_data, root=0)
else:
    # Other models receive the initial message
    initial_data = comm.bcast(None, root=0)
    initial_message = initial_data["message"]
    conversation_history.append({"sender": "Model 0", "message": initial_message})

# Number of conversation turns
max_turns = 10
current_turn = initial_data["turn"]

# Main conversation loop
while current_turn < max_turns:
    # Determine which model's turn it is to respond
    speaking_rank = current_turn % size
    
    if rank == speaking_rank:
        # This model's turn to generate a response
        
        # Format conversation history as context for the model
        context = "\n".join([f"{entry['sender']}: {entry['message']}" for entry in conversation_history])
        prompt = f"{context}\nModel {rank}:"
        
        # Generate response
        response = generate_response(prompt)
        print(f"Model {rank} generated: {response}")
        
        # Add to local conversation history
        conversation_history.append({"sender": f"Model {rank}", "message": response})
        
        # Broadcast response to all other models
        broadcast_data = {
            "sender": f"Model {rank}",
            "message": response,
            "turn": current_turn + 1
        }
        comm.bcast(broadcast_data, root=speaking_rank)
    else:
        # Wait to receive the response from the speaking model
        broadcast_data = comm.bcast(None, root=speaking_rank)
        conversation_history.append({
            "sender": broadcast_data["sender"],
            "message": broadcast_data["message"]
        })
    
    # Update turn counter
    current_turn = broadcast_data["turn"]
    
    # Small delay to keep things organized
    time.sleep(0.5)

# Save conversation transcript
os.makedirs("transcripts", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
transcript_path = f"transcripts/model_{rank}_{model_name.replace('/', '_')}_{timestamp}.json"

transcript_data = {
    "rank": rank,
    "model": model_name,
    "device": device,
    "conversation": conversation_history
}

with open(transcript_path, "w") as f:
    json.dump(transcript_data, f, indent=4)

print(f"Process {rank} completed. Conversation transcript saved to {transcript_path}")

# Optional: If you want all conversations to be collected at rank 0
if rank != 0:
    comm.send(transcript_data, dest=0)
    
if rank == 0:
    all_transcripts = [transcript_data]
    for i in range(1, size):
        all_transcripts.append(comm.recv(source=i))
    
    # Save complete conversation with all model perspectives
    complete_path = f"transcripts/complete_conversation_{timestamp}.json"
    with open(complete_path, "w") as f:
        json.dump(all_transcripts, f, indent=4)
    print(f"Complete conversation from all perspectives saved to {complete_path}")