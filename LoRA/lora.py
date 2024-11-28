# Script that will be used for LoRA training and save the model locally for further evaluation

#Imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import  load_dataset, load_from_disk, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
import time
import json
import os
import sys

# Constants
MODELPATH = "ModelMistral"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
PROCESSED_DIR = "Baseline/processed_datasets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R = 1

arg = sys.argv[1]
isMath = arg == "Math"

# Load the model and tokenizer from the local directory if available and not empty
""""
if os.path.exists(MODELPATH) and os.listdir(MODELPATH):
    print(f"Loading model from {MODELPATH}...")
    model = AutoModelForCausalLM.from_pretrained(MODELPATH)
    tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
else:
    print(f"Downloading model from {MODEL_NAME}...")
    # Used in HPC
    userToken = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if userToken:
        print("Token found.")
        login(token=userToken)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        # Save the model and tokenizer locally
    print(f"Saving model to {MODELPATH}...")
    model.save_pretrained(MODELPATH)
    tokenizer.save_pretrained(MODELPATH)
        
# Move the model to the GPU if available    
model.to(DEVICE)
"""

def load_processed_datasets(IsMath: bool):
    # Check if processed datasets exist locally
    if os.path.exists(f"{PROCESSED_DIR}/linguistic") and os.path.exists(f"{PROCESSED_DIR}/meta_math"):
        print("Loading processed datasets from local disk...")
        # Load the entire DatasetDict (not just individual splits)
        if isMath:
            meta_math_dataset = DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math")
            return meta_math_dataset
        else:
            linguistic_dataset = DatasetDict.load_from_disk(f"{PROCESSED_DIR}/linguistic")
            return linguistic_dataset
    else:
        raise Exception("Processed datasets not found")

processed_data = load_processed_datasets(isMath)
print(processed_data)
print(processed_data["train"][0])

# Tokenize datasets
def tokenize_data(dataDict, isMath):
    if isMath:
        inputs = dataDict["query"]
        outputs = dataDict["response"]

        inputs = tokenizer(examples["query"], max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
        outputs = tokenizer(examples["response"], max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
    else:
        inputs = dataDict["paragraph_generation_prompt"]
        outputs = dataDict["claude_summary"]

        inputs = tokenizer(examples["paragraph_generation_prompt"], max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
        outputs = tokenizer(examples["claude_summary"], max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": outputs["input_ids"]}

tokenized_train_data = tokenize_data(processed_data["train"], isMath)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=R,
    lora_alpha=R * 2,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

#model = get_peft_model(model, lora_config)

# Training arguments with early stopping
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_dir = f"LoRA_FineTuned_{'Math' if isMath else 'Linguistic'}_{timestamp}"

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    num_train_epochs=10,  # Set maximum number of epochs
    logging_dir=f"{output_dir}/logs",
    logging_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,  # Load the best model
    report_to="none",
    metric_for_best_model="eval_loss",  # Use evaluation loss for early stopping
    greater_is_better=False,  # Lower loss is better
)

# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2  # Stop if validation loss does not improve for 2 consecutive epochs
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback],  # Add the early stopping callback
)

# Train and save model
trainer.train()
model.save_pretrained(f"{output_dir}/final_model")
tokenizer.save_pretrained(f"{output_dir}/final_model")

print(f"Model fine-tuned and saved to {output_dir}/final_model")



    