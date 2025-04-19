"""
This is a basic, barebones boilerplate implementation of 
Reinforcement Learning (RL) using Proximal Policy Optimization (PPO) for a chatbot model.
"""

"""
Psuedocode
"""
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

MODEL_NAME = "microsoft/phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, 
                                          cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/")

class ExamDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == "__main__":
    # Load the dataset
    with open("sec-exam.json", "r") as f:
        data = json.load(f)
    
    for question in data:
        print(question)
        student_system_prompt = "Think of 5 questions that a student might ask in order to learn enough about the topic to answer the question:"
        student_prompt = question["question"]

        message = {"role": "user", "content": student_system_prompt + student_prompt}

        tokenizer.apply_chat_template(message)

