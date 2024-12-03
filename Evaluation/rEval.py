# Imports
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import DatasetDict
from peft import PeftModel

# Paths and constants
MODELPATH = "ModelMistral"
PROCESSED_DIR = "LoRA/tokenized_datasets"
LoRA_PATH = "LoRA_FineTuned_Math_R:4/final_model"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Load the model and tokenizer
if os.path.exists(MODELPATH) and os.listdir(MODELPATH):
    print(f"Loading model from {MODELPATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODELPATH)
    tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
else:
    raise FileNotFoundError(f"Model {MODELPATH} not found or directory is empty.")

# Load and apply LoRA layers
if os.path.exists(LoRA_PATH):
    print(f"Loading LoRA model from {LoRA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LoRA_PATH)
else:
    raise FileNotFoundError(f"LoRA model {LoRA_PATH} not found.")

# Move the model to the appropriate device
model.to(device)
model.eval()

# Load tokenized datasets
tokenized_data_path = os.path.join(PROCESSED_DIR, "meta_math_tokenized")
if os.path.exists(tokenized_data_path):
    print(f"Loading tokenized datasets from {tokenized_data_path}...")
    tokenized_data = DatasetDict.load_from_disk(tokenized_data_path)
else:
    raise FileNotFoundError(f"Tokenized datasets not found at {tokenized_data_path}.")

# Select a sample from the dataset
sample = tokenized_data["test"][0]

# Prepare input for the model
print("Preparing input for the model...")
inputs = {
    "input_ids": torch.tensor([sample["input_ids"]], device=device),
    "attention_mask": torch.tensor([sample["attention_mask"]], device=device),
}

# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate model response
print("Generating response...")
outputs = model.generate(
    **inputs,
    max_new_tokens=520,
    do_sample=True,
    temperature=0.7,  # Adjust for randomness
    top_k=50,         # Top-k sampling for diversity
    top_p=0.9,        # Nucleus sampling
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

base_response = base_model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)
print(tokenizer.decode(base_response[0], skip_special_tokens=True))

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Response:")
print(response)
