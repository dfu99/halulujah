"""
This is a basic, barebones boilerplate implementation of 
Reinforcement Learning (RL) using Proximal Policy Optimization (PPO) for a chatbot model.
"""

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

MODEL_NAME = "microsoft/phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, 
                                          cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/")
    
if __name__ == "__main__":
    # Load the dataset
    with open("sec-exam-test.json", "r") as f:
        data = json.load(f)

    # 

