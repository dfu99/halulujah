"""
This sets up an example conversation between multiple models using MPI.

We can check the results again in phi-3.5-workflows repository src/dialog/transcripts
"""

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
    "microsoft/Phi-3.5-mini-instruct"
]
model_name = model_names[rank % len(model_names)]

print(f"Process {rank} loading model {model_name} on {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name,
    cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/", 
    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, 
    cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/",
    trust_remote_code=True, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)

def generate_response(conversation_history):
    """Generate a response from the model given a prompt"""
    tokenized_chat = tokenizer.apply_chat_template(conversation_history, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    model_input = tokenized_chat.to(device)
    with torch.no_grad():
        outputs = model.generate(
            model_input,
            max_new_tokens=500, # Must be rather high to ensure the assistant answers rather than doing completions
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    response = tokenizer.decode(outputs[0][tokenized_chat.shape[1]:], skip_special_tokens=True)
    torch.cuda.empty_cache()  # Free up VRAM
    return response

class ConversationHistory:
    def __init__(self):
        
        self.system_msg = "".join([
            "Keep responses concise and to the point, typically under 100 words. ",
            "Don't make lists. ",
            "Avoid lists. ",
            "Be conversational. " ,
            "Focus only on directly answering the question without unnecessary elaboration. ",
            "Prioritize the most relevant information and omit supplementary details. ",
            "Use simple, direct language and avoid repetition. ",
            "Do not include examples unless specifically requested. ",
            "Play devil's advocate."
            ])
        self.history = [{"role": "system", "content": self.system_msg}]

    def append(self, role, message):
        self.history.append({"role": role
                            , "content": message})
    def get(self):
        return self.history
    
    def clear(self):
        self.history = [{"role": "system", "content": self.system_msg}]

    def __str__(self):
        return str(self.history)

    def __repr__(self):
        return str(self.history)
    
    def __len__(self):
        return len(self.history)
    
    def reverse_role(self, role):
        """
        Returns the opposite role of the given role
        """
        if role == "user":
            return "assistant"
        elif role == "assistant":
            return "user"
        else:
            return role

    def alternate_roles(self, role):
        """
        Ensures that the roles in the conversation history are alternating
        """
        for entry in reversed(self.history):
            if entry["role"] == "system":
                pass
            elif entry["role"] == role:
                pass
            else:
                entry["role"] = role
            role = self.reverse_role(role)

    def enforce_last_role(self):
        """
        Ensure the last role in the conversation history is the user
        """
        if self.history[-1]["role"] == "assistant":
            self.alternate_roles("user")
        return self.history

print("*"*50)
print(f"Process {rank} ready to start conversation")
print("*"*50)

# Create conversation history for each model
conversation_history = ConversationHistory()

# Start with the model at rank 0
if rank == 0:
    # Model 0 starts the conversation with a topic
    initial_message = "Let's debate. My stance is that AI is bad for society."
    initial_role ="user"
    conversation_history.append(initial_role, initial_message)
    initial_data = {"turn":0, "message": {"role": initial_role, "content": initial_message}}
    
    # Broadcast the initial message to all other processes
    comm.bcast(initial_data, root=0)
else:
    # Other models receive the initial message
    initial_data = comm.bcast(None, root=0)
    initial_chat = initial_data["message"]["content"]
    conversation_history.append("user", initial_chat)

# Set max number of alternating speaking turns
max_turns = 10
current_turn = initial_data["turn"]
print(f"Process {rank} starting at turn {current_turn}")

# Main conversation loop
while current_turn < max_turns:
    # Determine which model's turn it is to respond
    speaking_rank = current_turn % size

    if rank == speaking_rank:
        # This model's turn to generate a response
        print(f"Process {rank} generating response")
        
        # Format conversation history to always make the speaking model the assistant
        conversation_history.enforce_last_role()
        
        # Generate response
        response = generate_response(conversation_history.get())
        print(f"Model {rank} generated: {response}")
        
        # Add to local conversation history
        conversation_history.append("assistant", response)
        
        # Broadcast the updated chatlog to all other models
        # Increment turn counter
        broadcast_data = {
            "turn": current_turn + 1, 
            "message": {"role": "assistant", "content": response}
            }
        comm.bcast(broadcast_data, root=speaking_rank)
    else:
        print(f"Process {rank} waiting to receive response from model {speaking_rank}")
        # Wait to receive the response from the speaking model
        broadcast_data = comm.bcast(None, root=speaking_rank)
        # Add last message to conversation history
        message = broadcast_data["message"]
        conversation_history.append(message["role"], message["content"])
        
    # Update turn counter
    current_turn = broadcast_data["turn"]
    print(f"Process {rank} turn {current_turn} complete")

    # Add a small time delay to keep things organized
    time.sleep(0.5)

print("*"*50)
print(f"Process {rank} conversation complete")
print("*"*50)

# Save conversation transcript
os.makedirs("transcripts", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
transcript_path = f"transcripts/model_{rank}_{model_name.replace('/', '_')}_{timestamp}.json"

transcript_data = {
    "rank": rank,
    "model": model_name,
    "device": device,
    "conversation": conversation_history.get()
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
