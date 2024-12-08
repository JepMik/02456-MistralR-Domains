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
LoRA_PATH = "New_test/FineTuned_Math_R4/lora_weights"

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
# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


questions = [
    "[INST]\n Provide the answer to the following question: \nIf the eldest sibling is currently 20 years old and the three siblings are born 5 years apart, what will be the total of their ages 10 years from now?\n[/INST]",
    "[INST] Alice has 8 apples and wants to divide them equally among her 4 friends. How many apples does each person get [/INST]",
    "[INST] If i can sleep 8 hours, how many hours are left in the day for studying? [/INST]"
]
# Loop through the questions
for idx, q in enumerate(questions, start=1):
    print(f"\nProcessing Question {idx}:\n{q}")

    # Tokenize the question
    inputs = tokenizer(q, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Base model responses
    print("\nGenerating 3 responses with the Base Model...")
    base_responses = []
    for _ in range(3):
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
        base_responses.append(tokenizer.decode(base_outputs[0], skip_special_tokens=True))

    # Apply LoRA weights
    print(f"\nLoading LoRA weights from {LoRA_PATH}...")
    if os.path.exists(LoRA_PATH):
        model = PeftModel.from_pretrained(base_model, LoRA_PATH).to(device)
        model.eval()
    else:
        raise FileNotFoundError(f"LoRA weights directory {LoRA_PATH} not found.")

    # LoRA-enhanced model responses
    print("\nGenerating 3 responses with the LoRA-Enhanced Model...")
    lora_responses = []
    for _ in range(3):
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
        lora_responses.append(tokenizer.decode(lora_outputs[0], skip_special_tokens=True))

    # Print and save responses
    print("\n=== Base Model Responses ===")
    for i, response in enumerate(base_responses, start=1):
        print(f"Response {i}: {response}")

    print("\n=== LoRA-Enhanced Model Responses ===")
    for i, response in enumerate(lora_responses, start=1):
        print(f"Response {i}: {response}")