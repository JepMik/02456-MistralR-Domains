from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login
from datasets import load_dataset, load_from_disk, DatasetDict
import pandas as pd
import json
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from alive_progress import alive_bar
import logging

# Used in HPC
userToken = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Hub login
if userToken:
    print("Token found.")
    login(token=userToken)

print("Script started...")


# Constants
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "Baseline/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELPATH = "ModelMistral"
PROCESSED_DIR = "Baseline/processed_datasets"
BATCH_SIZE = 8

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('alive_progress')

    
def create_prompt(prompt):
    """
    Create a formatted prompt for the model to generate a response.
    """
    return "[INST]\n" + prompt + "\n[/INST]"
      


def generate_responses(dataset, prompt_column, ground_truth_column, name_id, max_new_tokens=520):
    """
    Generate responses for a given dataset using the specified prompt column.
    Include the ground truth (claude_summary) in the results.
    """
    results = {}
    batch_prompts = []
    batch_ids = []
    batch_groundtruths = []
    current_batch = []

    with alive_bar(len(dataset)) as bar:
        for i, example in enumerate(dataset):
            # Extract the prompt, ground truth (claude_summary), and id
            prompt = example[prompt_column]
            groundtruth = example[ground_truth_column]
            if name_id == "question_id":
                id = example[name_id]
            else:
                id = example[name_id] + str(i)
            
            # Append to the batch
            batch_prompts.append(prompt)
            batch_ids.append(id)
            batch_groundtruths.append(groundtruth)
            
            # If batch is full, process the batch
            if len(batch_prompts) == BATCH_SIZE or i == len(dataset) - 1:
                # Format all prompts in the batch
                formatted_prompts = [create_prompt(p) for p in batch_prompts]
                
                tokenizer.pad_token = tokenizer.eos_token
                            
                # Tokenize the batch
                inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
                model.config.pad_token_id = tokenizer.eos_token_id
                
                # Get the max input length for the batch
                max_input_length = inputs['input_ids'].shape[1]
                max_length = 1024  
                remaining_tokens = max_length - max_input_length
                
                # Ensure we don't generate more than the remaining space
                max_new_tokens = min(remaining_tokens, max_new_tokens)
           
                # Generate model responses for the entire batch
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.2,
                    top_k=50,
                    top_p=0.02,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Decode the responses for the batch
                batch_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

                # Store results for each example in the batch
                for idx, generated_response in zip(range(len(batch_responses)), batch_responses):
                    id = batch_ids[idx]
                    groundtruth = batch_groundtruths[idx]
                    results[id] = {
                        "prompt": batch_prompts[idx],
                        "generated_response": generated_response,
                        "groundtruth": groundtruth
                    }

                # Reset for next batch
                batch_prompts = []
                batch_ids = []
                batch_groundtruths = []
            bar()

    return results


def load_or_process_datasets():
    # Check if processed datasets exist locally
    if os.path.exists(f"{PROCESSED_DIR}/linguistic") and os.path.exists(f"{PROCESSED_DIR}/meta_math"):
        print("Loading processed datasets from local disk...")
        # Load the entire DatasetDict (not just individual splits)
        linguistic_dataset = DatasetDict.load_from_disk(f"{PROCESSED_DIR}/linguistic")
        meta_math_dataset = DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math")
        print("Loaded datasets from local disk.")
    else:
        print("Loading datasets from Hugging Face and processing them...")
        # Load the raw datasets from Hugging Face, with caching handled automatically
        linguistic_dataset = load_dataset("DopeorNope/Linguistic_Calibration_Llama3.1")
        meta_math_dataset = load_dataset("meta-math/MetaMathQA")
        
        # Drop unnecessary columns for baseline: 

        # Get dataset sizes and subsample to match the smallest dataset
        min_size = min(len(linguistic_dataset['train']), len(meta_math_dataset['train']))
        linguistic_dataset['train'] = linguistic_dataset['train'].shuffle(seed=42).select(range(min_size))
        meta_math_dataset['train'] = meta_math_dataset['train'].shuffle(seed=42).select(range(min_size))

        # Split the datasets into train, validation, and test (80%, 10%, 10%)
        linguistic_dataset = split_dataset(linguistic_dataset)
        meta_math_dataset = split_dataset(meta_math_dataset)

        # Save processed datasets to disk (entire DatasetDict)
        if not os.path.exists(f"{PROCESSED_DIR}/linguistic"):
            linguistic_dataset.save_to_disk(f"{PROCESSED_DIR}/linguistic")
        if not os.path.exists(f"{PROCESSED_DIR}/meta_math"):
            meta_math_dataset.save_to_disk(f"{PROCESSED_DIR}/meta_math")
        print("Datasets saved to disk.")

    return linguistic_dataset, meta_math_dataset

def split_dataset(dataset):
    # Split the dataset into 80% train, 10% validation, 10% test
    split_ratio = [0.8, 0.1, 0.1]
    train_test_split = dataset['train'].train_test_split(test_size=split_ratio[1] + split_ratio[2], seed=42)
    val_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    
    # Store the splits back into a DatasetDict
    return DatasetDict({
        'train': train_test_split['train'],
        'val': val_test_split['train'],
        'test': val_test_split['test']
    })

# Load or process the datasets
linguistic_dataset, meta_math_dataset = load_or_process_datasets()
#print(linguistic_dataset['test'])
#print(linguistic_dataset['test'][0])


# Load the model and tokenizer from the local directory if available and not empty
if os.path.exists(MODELPATH) and os.listdir(MODELPATH):
    print(f"Loading model from {MODELPATH}...")
    model = AutoModelForCausalLM.from_pretrained(MODELPATH)
    tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
else:
    print(f"Downloading model from {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Save the model and tokenizer locally
    print(f"Saving model to {MODELPATH}...")
    model.save_pretrained(MODELPATH)
    tokenizer.save_pretrained(MODELPATH)
    
# Move the model to the GPU if available    
model.to(DEVICE)
    
print("Model loaded.")

# We only consider test for baseline evaluation
ling_test = linguistic_dataset["test"]
math_test = meta_math_dataset["test"]


# Decrease ling test to 2 for testing
#ling_test = ling_test.select(range(20))
#math_test = math_test.select(range(20))

#print("Ling Test: " + str(ling_test))
#print("Math Test: " + str(math_test))

# print all in ling_test
#for i in range(len(ling_test)):
#    print(ling_test[i])


linguistic_res = generate_responses(ling_test, prompt_column="paragraph_generation_prompt", ground_truth_column="claude_summary", name_id="question_id")
math_res = generate_responses(math_test, prompt_column="query", ground_truth_column="response", name_id="type")

#print("Results: " + str(linguistic_res))
#print("Results: " + str(math_res))

# Save linguistic results to disk as json   
with open(f"{OUTPUT_DIR}/linguistic_results.json", "w") as f:
    json.dump(linguistic_res, f)
    
with open(f"{OUTPUT_DIR}/math_results.json", "w") as f:
    json.dump(math_res, f)



 