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
            max_new_tokens=100,
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
    
    def flip_roles(self):
        for i in range(len(self.history)):
            if self.history[i]["role"] == "user":
                self.history[i]["role"] = "assistant"
            elif self.history[i]["role"] == "assistant":
                self.history[i]["role"] = "user"
            elif self.history[i]["role"] == "system":
                pass
            else:
                raise ValueError("Invalid role in conversation history.")
        return self.history
    
    def enforce_last_role(self):
        """Ensure the last role in the conversation history is the user"""
        if self.history[-1]["role"] == "assistant":
            self.flip_roles()
        return self.history

print("*"*50)
print(f"Process {rank} ready to start conversation")
print("*"*50)

# Create conversation history for each model
conversation_history = ConversationHistory()

# Set initial topic based on rank 0's model
if rank == 0:
    # Model 0 starts the conversation with a topic
    initial_message = "What do you think about the future of artificial intelligence."
    initial_role ="user"
    conversation_history.append(initial_role, initial_message)
    initial_data = {"turn":0, "chatlog": conversation_history.get()}
    
    # Broadcast the initial message to all other processes
    comm.bcast(initial_data, root=0)
else:
    # Other models receive the initial message
    initial_data = comm.bcast(None, root=0)
    initial_chat = initial_data["chatlog"][-1]["content"]
    conversation_history.append("user", initial_chat)

# Number of conversation turns
max_turns = 5
current_turn = initial_data["turn"]
print(f"Process {rank} starting at turn {current_turn}")
print(f"Process {rank} conversation history: {initial_data['chatlog']}")

# Main conversation loop
while current_turn < max_turns:
    # Determine which model's turn it is to respond
    speaking_rank = current_turn % size

    if rank == speaking_rank:
        # This model's turn to generate a response
        print(f"Process {rank} generating response")
        
        # Format conversation history as context for the model
        conversation_history.enforce_last_role()
        
        # Generate response
        response = generate_response(conversation_history.get())
        print(f"Model {rank} generated: {response}")
        
        # Add to local conversation history
        conversation_history.append("assistant", response)
        print(f"Added to conversation history: {conversation_history.get()}")
        
        # Broadcast response to all other models
        # Flip the roles of the conversation history before broadcasting
        # So that the assistant is the user in the next turn
        broadcast_data = {
            "turn": current_turn + 1, 
            "chatlog": conversation_history.get()
            }
        comm.bcast(broadcast_data, root=speaking_rank)
    else:
        print(f"Process {rank} waiting to receive response from model {speaking_rank}")
        # Wait to receive the response from the speaking model
        broadcast_data = comm.bcast(None, root=speaking_rank)
        # Add to conversation history
        message = broadcast_data["chatlog"][-1]
        conversation_history.append(message["role"], message["content"])
        
    # Update turn counter
    print(broadcast_data)
    print(type(broadcast_data))
    current_turn = broadcast_data["turn"]
    print(f"Process {rank} turn {current_turn} complete")
    print(f"Process {rank} conversation history: {broadcast_data['chatlog']}")

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
