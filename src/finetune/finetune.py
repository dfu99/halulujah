from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from huggingface_hub import ModelCard, ModelCardData, HfApi
from datasets import load_dataset
from jinja2 import Template
from trl import SFTTrainer, SFTConfig
import yaml
import torch
import os

# Configurations
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
NEW_MODEL_NAME = "Phi-3.5-EGNIVIA"
DATASET_NAME = "EGNIVIA-finetune-dataset-lg"
SPLIT = "train"
MAX_SEQ_LENGTH = 2048
num_train_epochs = 1
license = "apache-2.0"
learning_rate = 1.41e-5
per_device_train_batch_size = 4
gradient_accumulation_steps = 1

if torch.cuda.is_bf16_supported():
    print("Using supported bfloat16")
    compute_dtype = torch.bfloat16
else:
    print("Using supported float16")
    compute_dtype = torch.float16

# Load the model, tokenizer, and dataset
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, 
                                                 cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/",
                                                 trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, 
                                              cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/",
                                              trust_remote_code=True)

# Load and split dataset
dataset = load_dataset(f"datasets/{DATASET_NAME}", split="train")
train_size = int(len(dataset) * 0.9)

train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# Preprocess the dataset
def formatting_prompts_func(examples):
    """
    Format examples using chat template, compatible with SFTTrainer
    """
    texts = []
    
    for i in range(len(examples['question'])):
        messages = [
            {"role": "system", "content": examples['context'][i]},
            {"role": "user", "content": examples['question'][i]},
            {"role": "assistant", "content": examples['answer'][i]}
        ]
        
        # Note: We need to create the tokenizer outside this function
        # since we don't want to load it repeatedly for each batch
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        texts.append(formatted)
    
    return texts

# Define the training arguments
args = TrainingArguments(
    eval_strategy="steps",
    per_device_train_batch_size=7,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    max_steps=-1,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    output_dir=NEW_MODEL_NAME,
    optim="paged_adamw_32bit",
    lr_scheduler_type="linear",
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    deepspeed=None,
    ddp_find_unused_parameters=False
)

# Create SFTConfig
sft_config = SFTConfig(
    output_dir=NEW_MODEL_NAME,
    dataset_text_field="text",
    max_seq_length=128,
)

import os
os.environ["WANDB_API_KEY"] = "c5aa150de8d95fc12d9fe92220f638eb6917c74b"

# Start the fine-tuning process
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func
)
trainer.train()