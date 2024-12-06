# Imports
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import DatasetDict
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import pandas as pd

# Used in HPC
userToken = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Hub login
if userToken:
    print("Token found.")
    login(token=userToken)

print("Script started...")



# Paths and constants
MODELPATH = "ModelMistral"
PROCESSED_DIR = "LoRA/tokenized_datasets"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
LoRA_PATH = "New/FineTuned_Math_R8/lora_weights"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Load the base model and tokenizer
if os.path.exists(MODELPATH) and os.listdir(MODELPATH):
    print(f"Loading base model from {MODELPATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODELPATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
else:
    print(f"Downloading model from {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Save the model and tokenizer locally
    print(f"Saving model to {MODELPATH}...")
    base_model.save_pretrained(MODELPATH)
    tokenizer.save_pretrained(MODELPATH)
    
tokenized_data_path = os.path.join(PROCESSED_DIR, "meta_math_tokenized")
if os.path.exists(tokenized_data_path):
    print(f"Loading tokenized datasets from {tokenized_data_path}...")
    tokenized_data = DatasetDict.load_from_disk(tokenized_data_path)
else:
    raise FileNotFoundError(f"Tokenized datasets not found at {tokenized_data_path}.")

print("Tokenized data test.")
sample_input = "[INST] Provide the answer to the following question: Milly is trying to determine the duration of her study session... [/INST]"
tokenizedt = tokenizer(sample_input, return_tensors="pt")
print(tokenizedt)

# Select a sample from the dataset
sample = tokenized_data["test"][0]
print("Sample data:")
print(sample)

if "input_ids" not in sample or "attention_mask" not in sample:
    raise ValueError("Sample data is missing `input_ids` or `attention_mask`.")

# Prepare input for the model
print("Preparing input for the model...")
inputs = {
    "input_ids": torch.tensor([sample["input_ids"]], device=device),
    "attention_mask": torch.tensor([sample["attention_mask"]], device=device),
}

print("Input prepared.")
print(inputs)

# Generate response using the base model
print("Generating response with the base model...") # Reload base model to avoid LoRA interference
base_outputs = base_model.generate(
    **inputs,
    max_new_tokens=520,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.02,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Load and apply LoRA layers
if os.path.exists(LoRA_PATH):
    print(f"Loading LoRA weights from {LoRA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LoRA_PATH).to(device)
    model.eval()
else:
    raise FileNotFoundError(f"LoRA weights directory {LoRA_PATH} not found.")


# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate response using the LoRA-enhanced model
print("Generating response with LoRA-enhanced model...")
lora_outputs = model.generate(
    **inputs,
    max_new_tokens=520,
    do_sample=True,
    temperature=0.2,
    top_k=50,
    top_p=0.02,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)



# Decode and print both responses
lora_response = tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)

print("\n=== Base Model Response ===")
print(base_response)

print("\n=== LoRA-Enhanced Model Response ===")
print(lora_response)

# Optional: Save both responses for comparison
output_dir = "outputs"
