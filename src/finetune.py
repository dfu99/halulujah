from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from huggingface_hub import ModelCard, ModelCardData, HfApi
from datasets import load_dataset
from jinja2 import Template
from trl import SFTTrainer
import yaml
import torch

# Configurations

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
NEW_MODEL_NAME = "Phi-3.5-EGNIVIA"
DATASET_NAME = "EGNIVIA-finetune-dataset"
# MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
# NEW_MODEL_NAME = "opus-samantha-phi-3-mini-4k"
# DATASET_NAME = "macadeliccc/opus_samantha"
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

# Load the model, tokenizer, and dataset
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, 
                                                 cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/",
                                                 trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, 
                                              cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/",
                                              trust_remote_code=True)
dataset = load_dataset(DATASET_NAME, split="train")

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
            add_generation_prompt=True
        )
        
        texts.append(formatted)
    
    return {"text": texts}

# Define the training arguments
args = TrainingArguments(
evaluation_strategy="steps",
per_device_train_batch_size=7,
gradient_accumulation_steps=4,
gradient_checkpointing=True,
learning_rate=1e-4,
fp16 = not torch.cuda.is_bf16_supported(),
bf16 = torch.cuda.is_bf16_supported(),
max_steps=-1,
num_train_epochs=3,
save_strategy="epoch",
logging_steps=10,
output_dir=NEW_MODEL_NAME,
optim="paged_adamw_32bit",
lr_scheduler_type="linear")

trainer = SFTTrainer(
model=model,
args=args,
train_dataset=dataset,
dataset_text_field="text",
max_seq_length=128,
formatting_func=formatting_prompts_func
)
trainer.train()