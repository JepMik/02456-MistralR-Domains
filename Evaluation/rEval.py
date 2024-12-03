# Imports
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from huggingface_hub import login
from datasets import load_dataset, load_from_disk, DatasetDict
import pandas as pd
import json
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from alive_progress import alive_bar
import logging
from peft import PeftModel

data_path = "LoRA/tokenized_datasets"

## Model dictionary
MODELPATH = "ModelMistral"
PROCESSED_DIR = "LoRA/tokenized_datasets"

lora_dict = {
                "MathR1": "LoRA_FineTuned_Math_R:1/final_model",
                "MathR4": "LoRA_FineTuned_Math_R:4/final_model",
            }
wDir = "Evaluation/Results"
model = sys.argv[1]
data_ptr = "meta_math_tokenized" if "Math" in model else "linguistic_tokenized"

## Load the model and tokenizer from the local directory if available and not empty
if os.path.exists(MODELPATH) and os.listdir(MODELPATH):
    print(f"Loading model from {MODELPATH}")
    model = AutoModelForCausalLM.from_pretrained(MODELPATH)
    tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
else:
    raise Exception(f"Model {MODELPATH} not found in the local directory")


# Apply the LoRA layers to the base model
model_with_lora = PeftModel.from_pretrained(model, lora_dict["MathR4"])

# Move to device (e.g., GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_with_lora.to(device)

# Load tokenized datasets
if os.path.exists(f"{PROCESSED_DIR}/linguistic_tokenized") and os.path.exists(f"{PROCESSED_DIR}/meta_math_tokenized"):
        print("Loading processed datasets from local disk...")
        tokenized_data = DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math_tokenized")
else:
    raise Exception("Tokenized datasets not found")


# Try and pass the first example to the model print outcome
sample = tokenized_data["test"][0]

# Decode the input and run it through the model
print("Preparing input for the model...")
# Convert input data into a dictionary
inputs = {
    "input_ids": torch.tensor([sample["input_ids"]], device=device),
    "attention_mask": torch.tensor([sample["attention_mask"]], device=device),
}

print(sample)

# Generate model responses
outputs = model_with_lora.generate(
    **inputs,
    max_new_tokens=520,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.2,
    top_k=50,
    top_p=0.02,
    eos_token_id=tokenizer.eos_token_id,
)

# Ensure the tokenizer has a pad token to handle padding if needed
tokenizer.pad_token = tokenizer.eos_token

# Generate model responses for the entire batch
outputs = model.generate(
    **inputs, 
    max_new_tokens=520,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.2,
    top_k=50,
    top_p=0.02,
    eos_token_id=tokenizer.eos_token_id,
                )

# Decode the responses for the batch
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated response: {response}")


