from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from huggingface_hub import ModelCard, ModelCardData, HfApi
from datasets import load_dataset
from jinja2 import Template
from trl import SFTTrainer
import yaml
import torch
import json, os

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
NEW_MODEL_NAME = "Phi-3.5-mini-CXYZ"
DATASET_NAME = "cxyz"
SPLIT = "train"
MAX_SEQ_LENGTH = 2048
num_train_epochs = 1
license = "apache-2.0"
learning_rate = 1.41e-5
per_device_train_batch_size = 4
gradient_accumulation_steps = 1

if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float16


def setup_distributed():
    """Setup distributed training"""
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # SLURM provides topology information
        hostnames = os.environ["SLURM_JOB_NODELIST"]
        
        # Initialize the distributed environment
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        return local_rank
    else:
        return 0

def load_dataset(file_path):
    """Load and preprocess the dataset."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Format data into instruction format
    formatted_data = []
    for item in data:
        # Create instruction format
        formatted_text = (
            f"Context: {item['context']}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )
        formatted_data.append({"text": formatted_text})
    
    return Dataset.from_list(formatted_data)

def tokenize_function(examples, tokenizer):
    """Tokenize the texts with appropriate padding and truncation."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

def main():
    # Setup distributed training
    local_rank = setup_distributed()

    # Initialize model and tokenizer
    model_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/finetune/")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},  # Map to local GPU
        use_cache=False,              # Disable KV cache during training
        gradient_checkpointing=True,   # Enable gradient checkpointing
        cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/finetune/"
    )

    # Load and preprocess dataset
    # Only load dataset on main process for distributed training
    if local_rank == 0 or not dist.is_initialized():
        dataset = load_dataset("dataset.json")
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    else:
        split_dataset = None
    
    # Configure training arguments for distributed training
    training_args = TrainingArguments(
        output_dir="./phi-ft-results",
        num_train_epochs=3,
        per_device_train_batch_size=2,      # Reduced batch size
        per_device_eval_batch_size=2,       # Reduced batch size
        gradient_accumulation_steps=4,      # Accumulate gradients
        warmup_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=True,                         # Enable mixed precision training
        optim="adamw_torch_fused",         # Use fused optimizer
        ddp_find_unused_parameters=False,   # DDP optimization
        local_rank=local_rank,             # Set local rank for distributed
        dataloader_num_workers=4,          # Parallel data loading
    )
    
    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"] if split_dataset else None,
        eval_dataset=split_dataset["test"] if split_dataset else None,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the model only on the main process
    if local_rank == 0 or not dist.is_initialized():
        trainer.save_model("/storage/home/hcoda1/6/dfu71/scratch/mymodels/phi-ft-final")
        tokenizer.save_pretrained("/storage/home/hcoda1/6/dfu71/scratch/mymodels/phi-ft-final")

if __name__ == "__main__":
    main()