# This script generates answers to prompts with different versions of LoRA-enhanced models.
# Saves responses and ground truths to a file for evaluation.

# Imports
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import DatasetDict
from peft import PeftModel
from huggingface_hub import login
import pandas as pd

# Constants
MODELPATH = "ModelMistral"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSED_DIR = "Baseline/processed_datasets"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 520
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Used in HPC
userToken = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Hub login
if userToken:
    print("Token found.")
    login(token=userToken)


# Data configuration
data_dict = {
    "Math": [ "FineTuned_Math_R1","FineTuned_Math_R4","FineTuned_Math_R8", "FineTuned_Math_R16"],
    "Linguistic": ["FineTuned_Linguistic_R1","FineTuned_Linguistic_R4", "FineTuned_Linguistic_R8", "FineTuned_Linguistic_R16"],
    "CrossDomain": ["FineTuned_Math_R1", "FineTuned_Linguistic_R1"]
}


data_chosen = sys.argv[1]

# Load the base model and tokenizer
if os.path.exists(MODELPATH) and os.listdir(MODELPATH):
    print(f"Loading base model from {MODELPATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODELPATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
else:
    print(f"Downloading model from {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Save the model and tokenizer locally
    print(f"Saving model to {MODELPATH}...")
    base_model.save_pretrained(MODELPATH)
    tokenizer.save_pretrained(MODELPATH)

print("Script started...")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to load processed datasets
def load_processed_datasets(isMath: bool):
    if os.path.exists(f"{PROCESSED_DIR}/linguistic") and os.path.exists(f"{PROCESSED_DIR}/meta_math"):
        print("Loading processed datasets from local disk...")
        if isMath:
            return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math")
        else:
            return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/linguistic")
    else:
        raise FileNotFoundError("Processed datasets not found in the specified directory.")
    
def load_all():
    return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math")["test"], DatasetDict.load_from_disk(f"{PROCESSED_DIR}/linguistic")["test"]

# Load the appropriate dataset
if data_chosen == "Math" or data_chosen == "Linguistic":
    dataset = load_processed_datasets(data_chosen == "Math")["test"]
    promptKey = "query" if data_chosen == "Math" else "paragraph_generation_prompt"
    groundTruthKey = "response" if data_chosen == "Math" else "claude_summary"
else:
    math_dataset, ling_dataset = load_all()
    crossDict = {"Math": {"prmptKey": "query", "grndTruthKey": "response"}, 
                 "Linguistic": {"prmptKey": "paragraph_generation_prompt", 
                                "grndTruthKey": "claude_summary"}}



result = {}
# Generate responses for each LoRA version
for version in data_dict[data_chosen]:
    print(f"Loading LoRA model: {version}...")
    if os.path.exists(f"LoraWeights/{version}/lora_weights"):
        continue
    LoRA_PATH = f"LoraWeights/{version}/lora_weights"
    lora_model = PeftModel.from_pretrained(base_model, LoRA_PATH).to(DEVICE)
    lora_model.eval()

    # Results key
    result[version] = {}

    # Batch processing
    batch_prompts = []
    batch_ground_truths = []
    batch_ids = []
    print(f"Generating responses for {version}...")

    for i, example in enumerate(dataset):
        print(f"Processing example {i + 1}/{len(dataset)}...", end="\r")
        # Prepare inputs
        input_text = example[promptKey]
        ground_truth = example[groundTruthKey]

        batch_prompts.append(input_text)
        batch_ground_truths.append(ground_truth)
        batch_ids.append(i)

        # Process when the batch is full or at the end
        if len(batch_prompts) == BATCH_SIZE or i == len(dataset) - 1:
            # Tokenize the batch
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=524).to(DEVICE)
                
            # Calculate max_new_tokens dynamically
            max_input_length = inputs['input_ids'].shape[1]
            max_length = 1024  # Model's maximum sequence length
            remaining_tokens = max_length - max_input_length
            adjusted_max_new_tokens = min(remaining_tokens, MAX_NEW_TOKENS)

            # Generate responses
            with torch.no_grad():
                outputs = lora_model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.2,
                    top_k=50,
                    top_p=0.02,
                    eos_token_id=tokenizer.eos_token_id,
                    )

            # Decode and store responses
            batch_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            for idx, response in enumerate(batch_responses):
                result[version][batch_prompts[idx]] = {
                    "response": response,
                    "ground_truth": batch_ground_truths[idx]
                }

            # Reset batch
            batch_prompts = []
            batch_ground_truths = []
            batch_ids = []

    # Delete the LoRA model to free memory
    del lora_model
    torch.cuda.empty_cache()


def CrossDomainResponses(base_model, dataset, promptKey, groundTruthKey, version):
    result = {}
   
    print(f"Loading LoRA model: {version}...")
    LoRA_PATH = f"LoraWeights/{version}/lora_weights"
    lora_model = PeftModel.from_pretrained(base_model, LoRA_PATH).to(DEVICE)
    lora_model.eval()

    # Batch processing
    batch_prompts = []
    batch_ground_truths = []
    batch_ids = []
    print(f"Generating responses for {version}...")

    for i, example in enumerate(dataset):
        print(f"Processing example {i + 1}/{len(dataset)}...", end="\r")
        # Prepare inputs
        input_text = example[promptKey]
        ground_truth = example[groundTruthKey]

        batch_prompts.append(input_text)
        batch_ground_truths.append(ground_truth)
        batch_ids.append(i)

        # Process when the batch is full or at the end
        if len(batch_prompts) == BATCH_SIZE or i == len(dataset) - 1:
            # Tokenize the batch
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=524).to(DEVICE)

            # Calculate max_new_tokens dynamically
            max_input_length = inputs['input_ids'].shape[1]
            max_length = 1024  # Model's maximum sequence length
            remaining_tokens = max_length - max_input_length
            adjusted_max_new_tokens = min(remaining_tokens, MAX_NEW_TOKENS)

            # Generate responses
            with torch.no_grad():
                outputs = lora_model.generate(
                    **inputs,
                    max_new_tokens=adjusted_max_new_tokens,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.2,
                    top_k=50,
                    top_p=0.02,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode and store responses
            batch_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            for idx, response in enumerate(batch_responses):
                result[batch_prompts[idx]] = {
                    "response": response,
                    "ground_truth": batch_ground_truths[idx]
                }

            # Reset batch
            batch_prompts = []
            batch_ground_truths = []
            batch_ids = []

        # Delete the LoRA model to free memory
    del lora_model
    torch.cuda.empty_cache()
    return result

print(f"Generating responses started..")

""""
if data_chosen == "CrossDomain":
    result = {}
    for version in data_dict[data_chosen]:
        print(f"Loading LoRA model: {version}...")
        isMath = True if "Math" in version else False
        promptKey = crossDict["Linguistic"]["prmptKey"] if isMath else crossDict["Math"]["prmptKey"]
        groundTruthKey = crossDict["Linguistic"]["grndTruthKey"] if isMath else crossDict["Math"]["grndTruthKey"] 
        # Cross domain responses
        dataset = ling_dataset if isMath else math_dataset
        result[version] = CrossDomainResponses(base_model,dataset,promptKey,groundTruthKey, version)
else:
    result = math_ling_gen_responses(base_model,data_chosen,data_dict,dataset,promptKey,groundTruthKey)
"""


# Save the results to a file json file
output_path = f"Results/{data_chosen}1_4_8_16_results.json"
pd.DataFrame(result).to_json(output_path)
print(f"Results saved to {output_path}.")