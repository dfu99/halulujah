"""
Experiment 3: RL training on graded responses
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os

MODEL_NAME = "microsoft/phi-3.5-mini-instruct"
CACHE_DIR = "/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, 
                                           cache_dir=CACHE_DIR,
                                           trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
                                              cache_dir=CACHE_DIR,
                                              trust_remote_code=True).to(DEVICE)

# Load the data
with open("sec-exam-test.json", "r") as f:
    data = json.load(f)

# Initialize RL components

# Define the PPO training loop
def train_ppo(data, model, tokenizer):
    """
    Train the model using Proximal Policy Optimization (PPO).
    """
    # Placeholder for PPO training logic
    pass
    # Iterate over the dataset
    for question in data:
        # Get the model's response
        response = get_response(question)
        
        # Get the reward from the grading function
        reward = compare_responses(question)
        
        # Update the model using PPO
        update_model_with_ppo(response, reward)
        # Save the model periodically
        if i % 100 == 0:
            model.save_pretrained(f"model_checkpoint_{i}.pt")
            tokenizer.save_pretrained(f"tokenizer_checkpoint_{i}.pt")
    # Save the model at the end
    model.save_pretrained("final_model.pt")
    tokenizer.save_pretrained("final_tokenizer.pt")


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
    reward_model=reward_model,
    processing_class=tokenizer,
    train_dataset=dataset
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