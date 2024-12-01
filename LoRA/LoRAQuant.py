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

# Constants
MODELPATH = "ModelMistral"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
PROCESSED_DIR = "LoRA/tokenized_datasets"
R = 1

# Initialize the Accelerator
accelerator = Accelerator()

# Device setup
DEVICE = accelerator.device
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

arg = sys.argv[1]
isMath = arg == "Math"

# Load the model and tokenizer
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        load_in_8bit=True,  # Enable quantization
        device_map="auto",  # Auto-distribute model layers
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"Saving model to {MODELPATH}...")
    model.save_pretrained(MODELPATH)
    tokenizer.save_pretrained(MODELPATH)

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

# Prepare the model for LoRA
model = get_peft_model(model, lora_config)

# Prepare the model and datasets with the Accelerator
model, tokenized_data["train"], tokenized_data["val"] = accelerator.prepare(
    model, tokenized_data["train"], tokenized_data["val"]
)

output_dir = f"LoRAQuant_FineTuned_{'Math' if isMath else 'Linguistic'}_R:{R}"

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=1,  # Adjust based on available GPU memory
    num_train_epochs=10,
    logging_dir=f"{output_dir}/logs",
    logging_steps=50,
    save_total_limit=2,
    fp16=True,
    load_best_model_at_end=True,
    report_to="none",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=8,
)

# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

# Trainer setup
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

# Train and save the model
trainer.train()
end = time.time()
print(f"Training time: {end - start} seconds")

# Save training time, model, and tokenizer
with open(f"{output_dir}/training_time.txt", "w") as f:
    f.write(f"Training time: {end - start} seconds")
model.save_pretrained(f"{output_dir}/final_model")
tokenizer.save_pretrained(f"{output_dir}/final_model")

print(f"Model fine-tuned and saved to {output_dir}/final_model")
