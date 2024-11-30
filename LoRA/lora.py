# Script that will be used for LoRA training and save the model locally for further evaluation

#Imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import  load_dataset, load_from_disk, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
import time
import json
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Constants
MODELPATH = "ModelMistral"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
PROCESSED_DIR = "LoRA/tokenized_datasets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R = 1

arg = sys.argv[1]
isMath = arg == "Math"

# Load the model and tokenizer from the local directory if available and not empty
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

torch.cuda.empty_cache()
def load_tokenized_datasets(IsMath: bool):
    # Check if processed datasets exist locally

    print(f"{PROCESSED_DIR}/meta_math_tokenized" if IsMath else f"{PROCESSED_DIR}/linguistic_tokenized")
    if os.path.exists(f"{PROCESSED_DIR}/linguistic_tokenized") and os.path.exists(f"{PROCESSED_DIR}/meta_math_tokenized"):
        print("Loading processed datasets from local disk...")
        # Load the entire DatasetDict (not just individual splits)
        if isMath:
            meta_math_dataset = DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math_tokenized")
            return meta_math_dataset
        else:
            linguistic_dataset = DatasetDict.load_from_disk(f"{PROCESSED_DIR}/linguistic_tokenized")
            return linguistic_dataset
    else:
        raise Exception("Tokenized datasets not found")

tokenized_data = load_tokenized_datasets(isMath)
# Set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token
# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=R,
    lora_alpha=R * 2,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

output_dir = f"LoRA_FineTuned_{'Math' if isMath else 'Linguistic'}_R:{R}"
# Training arguments with early stopping
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    num_train_epochs=10,  # Set maximum number of epochs
    logging_dir=f"{output_dir}/logs",
    logging_steps=50,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    load_best_model_at_end=True,  # Load the best model
    report_to="none",
    metric_for_best_model="eval_loss",  # Use evaluation loss for early stopping
    greater_is_better=False,  # Lower loss is better
    gradient_accumulation_steps=8,
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
    eval_dataset=tokenized_data["val"],
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback],  # Add the early stopping callback
)
print("Starting training...")
# Train and save model
start = time.time()
trainer.train()
end = time.time()
print(f"Training time: {end - start} seconds")


# Save time taken for training, model and tokenizer
with open(f"{output_dir}/training_time.txt", "w") as f:
    f.write(f"Training time: {end - start} seconds")
model.save_pretrained(f"{output_dir}/final_model")
tokenizer.save_pretrained(f"{output_dir}/final_model")

print(f"Model fine-tuned and saved to {output_dir}/final_model")



    