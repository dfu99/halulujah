import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

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
    # Initialize model and tokenizer
    model_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/finetune/")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/storage/home/hcoda1/6/dfu71/scratch/.cache/finetune/"
    )
    
    # Load and preprocess dataset
    dataset = load_dataset("dataset.json")
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset into train and validation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="./phi-ft-results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
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
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model("/storage/home/hcoda1/6/dfu71/scratch/mymodels/phi-ft-final")
    tokenizer.save_pretrained("/storage/home/hcoda1/6/dfu71/scratch/mymodels/phi-ft-final")

if __name__ == "__main__":
    main()