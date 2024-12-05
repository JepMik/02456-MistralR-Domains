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
LoRA_PATH = "FineTuned_Math_R4_Test/lora_weights"

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
    raise FileNotFoundError(f"Base model directory {MODELPATH} not found or is empty.")

# Load and apply LoRA layers
if os.path.exists(LoRA_PATH):
    print(f"Loading LoRA weights from {LoRA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LoRA_PATH).to(device)
    model.eval()
else:
    raise FileNotFoundError(f"LoRA weights directory {LoRA_PATH} not found.")

# Load tokenized datasets
tokenized_data_path = os.path.join(PROCESSED_DIR, "meta_math_tokenized")
if os.path.exists(tokenized_data_path):
    print(f"Loading tokenized datasets from {tokenized_data_path}...")
    tokenized_data = DatasetDict.load_from_disk(tokenized_data_path)
else:
    raise FileNotFoundError(f"Tokenized datasets not found at {tokenized_data_path}.")

# Select a sample from the dataset
sample = tokenized_data["test"][0]
if "input_ids" not in sample or "attention_mask" not in sample:
    raise ValueError("Sample data is missing `input_ids` or `attention_mask`.")

# Prepare input for the model
print("Preparing input for the model...")
inputs = {
    "input_ids": torch.tensor([sample["input_ids"]], device=device),
    "attention_mask": torch.tensor([sample["attention_mask"]], device=device),
}

# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate response using the LoRA-enhanced model
print("Generating response with LoRA-enhanced model...")
lora_outputs = model.generate(
    **inputs,
    max_new_tokens=520,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Generate response using the base model
print("Generating response with the base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODELPATH).to(device)  # Reload base model to avoid LoRA interference
base_outputs = base_model.generate(
    **inputs,
    max_new_tokens=520,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
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
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "base_model_response.txt"), "w") as f:
    f.write(base_response)

with open(os.path.join(output_dir, "lora_model_response.txt"), "w") as f:
    f.write(lora_response)
