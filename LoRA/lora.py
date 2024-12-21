## Script to fine-tune a pretrained model with LoRA adapter
## Usage: python lora.py <Math|Linguistic> <R>

# Imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset, load_from_disk, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from accelerate import Accelerator
import time
import json
import os
import sys
import matplotlib.pyplot as plt

# Constants
MODELPATH = "ModelMistral"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
PROCESSED_DIR = "LoRA/Working/tokenized_datasets"

# Initialize the Accelerator
accelerator = Accelerator()

# Device setup
DEVICE = accelerator.device
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

arg = sys.argv[1]
R = int(sys.argv[2]) # Number of LoRA layers

isMath = arg == "Math"

# Load tokenized datasets
def load_tokenized_datasets(IsMath: bool):
    print(f"{PROCESSED_DIR}/meta_math_tokenized" if IsMath else f"{PROCESSED_DIR}/linguistic_tokenized")
    if os.path.exists(f"{PROCESSED_DIR}/linguistic_tokenized") and os.path.exists(f"{PROCESSED_DIR}/meta_math_tokenized"):
        print("Loading processed datasets from local disk...")
        if isMath:
            return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math_tokenized")
        else:
            return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/linguistic_tokenized")
    else:
        raise Exception("Tokenized datasets not found")

# Load model and tokenizer and download if not found locally
def load_model():
    if os.path.exists(MODELPATH) and os.listdir(MODELPATH):
        print(f"Loading model from {MODELPATH}...")
        model = AutoModelForCausalLM.from_pretrained(MODELPATH)
        tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
    else:
        print(f"Downloading model from {MODEL_NAME}...")
        userToken = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if userToken:
            print("Token found.")
            login(token=userToken)

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print(f"Saving model to {MODELPATH}...")
        model.save_pretrained(MODELPATH)
        tokenizer.save_pretrained(MODELPATH)
    return model, tokenizer


print("Loading model and tokenizer...")
model, tokenizer = load_model()
tokenized_data = load_tokenized_datasets(isMath)


print("Starting freezing...")
# Freeze the base model weights
for param in model.parameters():
    param.requires_grad = False

print("Starting fine-tuning...")
print(f"Fine-tuning with R={R}...")

# Set the padding token to eos token
tokenizer.pad_token = tokenizer.eos_token

# Set the LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # Casual Language Modeling
    inference_mode=False,   
    r=R, # Rank of the adapter
    lora_alpha=R, # Alpha value for LoRA
    lora_dropout=0.1, # Dropout value for LoRA
    target_modules=["q_proj", "v_proj","o_proj","gate_proj"] # Target modules for LoRA adapter training
)
# Peft model to load the LoRA adapter
model = get_peft_model(model, lora_config)

# Prepare the model for training
model, tokenized_data["train"], tokenized_data["val"] = accelerator.prepare(
    model, tokenized_data["train"], tokenized_data["val"]
)

# Set the output directory
output_dir = f"FineTuned_{'Math' if isMath else 'Linguistic'}_R{R}"

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4, # Recommended learning rate for LoRA, decreases automatically
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_dir=f"{output_dir}/logs",
    logging_steps=100,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    report_to="none",
    metric_for_best_model="eval_loss", # Eval loss = CrossEntropyLoss for causal LM
    greater_is_better=False,
    gradient_accumulation_steps=8,
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2) # Early stopping callback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["val"],
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback],
)

print("Starting training...")
start = time.time()
trainer.train()
end = time.time()
print(f"Training time: {end - start} seconds")

# Save only the LoRA weights
lora_output_dir = f"LoraWeights/{output_dir}/lora_weights"
model.save_pretrained(lora_output_dir)
print(f"LoRA weights saved to {lora_output_dir}")

