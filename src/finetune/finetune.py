# Load all our packages
import sys
import logging

import datasets
from datasets import load_dataset
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Configurations

logger = logging.getLogger(__name__)

###################
# Hyper-parameters
###################
training_config = {
    "bf16": True,
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 50,
    "max_steps": -1,
    "output_dir": "models/checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
    "dataset_text_field": "text", # Moved from SFTTrainer arguments
    "packing": True, # Moved from SFTTrainer arguments
    "max_length": 2048,
    # "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
    # "deepspeed": "src/finetune/deepspeed_config.json",
    # "ddp_find_unused_parameters": False,  # Set to False for PEFT
    }

train_conf = SFTConfig(**training_config) # Changed to SFTConfig from TrainingArguments

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")


# For distributed training
# import torch.distributed as dist
# local_rank = int(os.environ.get("LOCAL_RANK", 0))
# torch.cuda.set_device(local_rank)
# dist.init_process_group(backend='nccl')

# Load the model, tokenizer, and dataset

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
NEW_MODEL_NAME = "Phi-3.5-EGNIVIA-lg"
DATASET_NAME = "datasets/EGNIVIA-finetune-dataset-lg"
cache_dir = "/storage/home/hcoda1/6/dfu71/scratch/.cache/huggingface/"

model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map=None
)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs,
                                                 cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                              cache_dir=cache_dir)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

##################
# Data Processing
##################
# Load and split dataset
dataset = load_dataset(DATASET_NAME, split="train")
train_size = int(len(dataset) * 0.9)

train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))
column_names = list(train_dataset.features)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

def convert_example(example):
    user_content = f"{example['context']}\n\nQuestion: {example['question']}"
    assistant_content = example["answer"]
    example["messages"] = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    return example


train_dataset = train_dataset.map(convert_example, remove_columns=column_names)
test_dataset = test_dataset.map(convert_example, remove_columns=column_names)
column_names = list(train_dataset.features)

def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

processed_train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to train_sft",
)

processed_test_dataset = test_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=10,
    remove_columns=column_names,
    desc="Applying chat template to test_sft",
)
###########
# Training
###########

import os
os.environ["WANDB_API_KEY"] = "c5aa150de8d95fc12d9fe92220f638eb6917c74b"

trainer = SFTTrainer(
    model=model,
    args=train_conf,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    processing_class=tokenizer
)
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


#############
# Evaluation
#############
tokenizer.padding_side = 'left'
metrics = trainer.evaluate()
metrics["eval_samples"] = len(processed_test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# ############
# # Save model
# ############
trainer.save_model(train_conf.output_dir)