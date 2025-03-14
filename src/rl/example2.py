"""
This code was generated with the assistance of Claude.ai
The purpose of this code is to demonstrate the state of a model before and after Reinforcement Learning with
Human Feedback (RLHF) training.
"""

import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)

CACHE_DIR = "/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/"


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model parameters
MODEL_NAME = "microsoft/phi-3.5-mini-instruct"
OUTPUT_DIR = "/storage/home/hcoda1/6/dfu71/scratch/halulujah/phi-3.5-rlhf-output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the base model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Sample prompts for evaluating the model before and after training
EVALUATION_PROMPTS = [
    "Tell me about EGNIVIA's business model.",
    "Advise me whether I should invest in EGNIVIA.",
    "Rate EGNIVIA as a buy, sell, or hold.",
    "What is EGNIVIA's target market?",
    "What are potential risks for EGNIVIA?",
]

SYSTEM_PROMPT = (
    "You are an analyst at an interview where the prospective employer is asking you about a company that is"
    "not within your existing knowledge. ",
    "They intend to test your thought process and approach in understanding a new company. ",
    "The name of the company is EGNIVIA. ",
    "Do not ramble. Rambling is heavily penalized. ",
    "Do not make things overly complex. ",
    "Focus on factual and rational analysis. ",
    "Do not make assumptions without evidence. ",
    "Do not provide personal experiences. ",
    "Do not provide anecdotes. "
    )

def evaluate_model(model, tokenizer, prompts):
    """Generate responses for the prompts using the given model."""
    results = []
    
    for prompt in prompts:
        # Use apply_chat_template to format the conversation
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": prompt}]
        formatted_input = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            return_tensors="pt",
            add_generation_prompt=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=formatted_input,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        # Extract just the assistant's response by getting text after the input
        response = tokenizer.decode(outputs[0][formatted_input.shape[1]:], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})
    
    return results

print("\n=== Evaluating model before RLHF training ===")
before_responses = evaluate_model(model, tokenizer, EVALUATION_PROMPTS)
for i, result in enumerate(before_responses):
    print(f"\nPrompt {i+1}: {result['prompt']}")
    print(f"Response: {result['response']}")
# Create a synthetic dataset for RLHF training
# In a real scenario, you would use actual human feedback data
# Create a more straightforward dataset for RLHF training
class SyntheticDataset(torch.utils.data.Dataset):
    """A simple PyTorch dataset for RLHF training with synthetic prompts."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompts = [
            "Explain Apple's business model.",
            "What are the risks of investing in Tesla?",
            "What is the target market for Amazon's products?",
            "Describe the competitive landscape for Google.",
            "How does Facebook make money?",
            "What are the potential risks for investing in Bitcoin?",
            "Explain the concept of cloud computing.",
            "What are the advantages of renewable energy?",
            "How does Netflix's subscription model work?",
            "Describe the impact of artificial intelligence on healthcare."
            "What are the key features of the latest iPhone model?",
            "What are the benefits of using a VPN service?",
            "Explain the concept of blockchain technology.",
            "What are the risks of investing in cryptocurrencies?",
            "How does Amazon's Prime membership work?",
            "What are the advantages of electric vehicles?",
            "Describe the impact of social media on society.",
            "What are the key features of the latest Android smartphone?",
            "What are the benefits of cloud storage services?",
            "Explain the concept of machine learning.",
        ]
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        # Only format the prompt when accessing it, not when creating the dataset
        messages = [{"role": "user", "content": prompt}]
        return {
            "prompt": prompt,
            "formatted_prompt": self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
        }

# Create a dataloader directly with PyTorch
def create_synthetic_dataloader(tokenizer, batch_size=4):
    """Create a PyTorch DataLoader with synthetic prompts."""
    dataset = SyntheticDataset(tokenizer)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

# Define a reward model (in a real scenario, you would train this on human feedback)
class SimpleRewardModel:
    """A simple reward model that evaluates responses based on basic heuristics."""
    
    def __init__(self):
        self.positive_keywords = [
            "detailed", "step by step", "example", "comprehensive", 
            "clear", "concise", "helpful", "understand", "simple"
        ]
        self.negative_keywords = [
            "sorry", "cannot", "don't know", "unclear", "confusing",
            "insufficient", "incomplete", "vague"
        ]
    
    def compute_reward(self, responses):
        """Compute rewards for a batch of responses."""
        rewards = []
        
        for response in responses:
            # Basic heuristics for scoring
            score = 0.5  # Start with a neutral score
            
            # Check length - we want reasonably detailed responses
            if len(response.split()) > 50:
                score += 0.1
            
            # Check for positive keywords
            for keyword in self.positive_keywords:
                if keyword.lower() in response.lower():
                    score += 0.05
            
            # Check for negative keywords
            for keyword in self.negative_keywords:
                if keyword.lower() in response.lower():
                    score -= 0.05
            
            # Check for coherent structure (paragraphs)
            if len(response.split('\n\n')) > 1:
                score += 0.1
            
            # Normalize to 0-1 range
            score = max(0.0, min(1.0, score))
            rewards.append(score)
        
        return torch.tensor(rewards)

# Initialize RLHF components
print("\n=== Setting up RLHF training ===")

# Set up the model with a value head for PPO
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    device_map="auto"
)

# Create LoRA configuration for efficient fine-tuning
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA to the model
ppo_model = get_peft_model(ppo_model, peft_config)
ppo_model.print_trainable_parameters()

# Create a reference model (frozen copy of the original model)
ref_model = create_reference_model(ppo_model)

# Set up the PPO configuration
ppo_config = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=8,
    mini_batch_size=1,
    num_ppo_epochs=4,
    gradient_accumulation_steps=1,
    kl_coef=0.1,
    gamma=0.99,
    output_dir=OUTPUT_DIR
)

# Initialize the reward model
reward_model = SimpleRewardModel()

# Create a dataset for training
dataset = create_synthetic_dataloader(tokenizer)
print(f"Created synthetic dataset with {len(dataset)} examples")

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    args=ppo_config,
    model=ppo_model,
    ref_model=ref_model,
    processing_class=tokenizer,
    dataset=dataset,
)

# Training loop
print("\n=== Starting RLHF training ===")
for epoch in range(3):  # Small number of epochs for demonstration
    print(f"\nEpoch {epoch+1}/3")
    
    for batch_idx, batch in enumerate(ppo_trainer.dataloader):
        # Generate responses using the current policy
        query_tensors = [tokenizer(prompt, return_tensors="pt").input_ids.to(device) for prompt in batch["prompt"]]

        # Now extract each prompt's original length to properly identify generated content
        input_lengths = [tensor.shape[1] for tensor in query_tensors]
        
        # Use PPOTrainer.generate instead of respond_to_batch
        response_tensors = []
        for query in query_tensors:
            generation_output = ppo_trainer.generate(
                query,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            response_tensors.append(generation_output.squeeze(0))
        
        # Extract responses as text
        batch_responses = []
        for i, (query, response, input_length) in enumerate(zip(query_tensors, response_tensors, input_lengths)):
            # Get the full generated text
            full_text = tokenizer.decode(response, skip_special_tokens=True)

            # Get the original input text
            input_text = tokenizer.decode(query.squeeze(), skip_special_tokens=True)

            # Extract just the assistant's response
            assistant_response = full_text[len(input_text):].strip()
            batch_responses.append(assistant_response)
        
        # Compute rewards
        rewards = reward_model.compute_reward(batch_responses)
        
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        
        print(f"  Batch {batch_idx+1}, Mean reward: {rewards.mean().item():.4f}")
        
        # Print a sample response for visibility
        if batch_idx % 2 == 0:
            sample_idx = 0
            print(f"  Sample query: {batch['query'][sample_idx]}")
            print(f"  Sample response: {batch_responses[sample_idx][:100]}...")
            print(f"  Sample reward: {rewards[sample_idx].item():.4f}")

# Save the fine-tuned model
ppo_trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nSaved fine-tuned model to {OUTPUT_DIR}")

# Load the fine-tuned model for evaluation
print("\n=== Loading fine-tuned model for evaluation ===")
fine_tuned_model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    device_map="auto"
)

# Evaluate the fine-tuned model
print("\n=== Evaluating model after RLHF training ===")
after_responses = evaluate_model(fine_tuned_model, tokenizer, EVALUATION_PROMPTS)

# Compare before and after results
print("\n=== Comparison of model responses before and after RLHF training ===")
for i, (before, after) in enumerate(zip(before_responses, after_responses)):
    print(f"\nPrompt {i+1}: {before['prompt']}")
    print(f"Before RLHF: {before['response'][:150]}...")
    print(f"After RLHF: {after['response'][:150]}...")
    
    # Compute reward scores for comparison
    before_reward = reward_model.compute_reward([before['response']]).item()
    after_reward = reward_model.compute_reward([after['response']]).item()
    
    print(f"Before reward: {before_reward:.4f}")
    print(f"After reward: {after_reward:.4f}")
    print(f"Improvement: {(after_reward - before_reward):.4f}")

# Overall statistics
before_rewards = reward_model.compute_reward([r['response'] for r in before_responses])
after_rewards = reward_model.compute_reward([r['response'] for r in after_responses])

print("\n=== Overall RLHF impact ===")
print(f"Average reward before RLHF: {before_rewards.mean().item():.4f}")
print(f"Average reward after RLHF: {after_rewards.mean().item():.4f}")
print(f"Overall improvement: {(after_rewards.mean() - before_rewards.mean()).item():.4f}")