from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login
from datasets import load_dataset, load_from_disk, DatasetDict
import pandas as pd
import json
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

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

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token matches the stop token
        return input_ids[0, -1] == self.stop_token_id


def generate_responses(dataset, prompt_column, ground_truth_column, name_id, max_new_tokens=150):
    """
    Generate responses for a given dataset using the specified prompt column.
    Include the ground truth (claude_summary) in the results.
    """
    
    
    
    # Encode the stop token
    stop_token_id = tokenizer.encode("###END", add_special_tokens=False)[0]
    # Define stopping criteria
    stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])
    results = {}
    for i, example in enumerate(dataset):
        # Extract the prompt, ground truth (claude_summary), and id
        prompt = example[prompt_column]
        groundtruth = example[ground_truth_column]
        if name_id == "question_id":
            id = example[name_id]
        else:
            id = example[name_id] + str(i)
        
        # Format the prompt
        formatted_prompt = f"###Query:\n {prompt}\n###Answer:\n"
        
        # Tokenize the input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
        
        # Generate model response
        output = model.generate(**inputs, 
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                                temperature=0.2,
                                top_k=50,
                                top_p=0.02,
                                eos_token_id=tokenizer.eos_token_id,
                                )
        # Decode the response
        generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Try to split the generated response to remove the redundant part answer token
        try:
            generate_responses_split = generated_response.split("###Answer")
            generate_responses = generate_responses_split[0] + generate_responses_split[1]
        except:
            generate_responses = generated_response
        
        
        # Append results with id as key
        results[id] = {
            "prompt": prompt,
            "generated_response": generated_response,
            "groundtruth": groundtruth
        }
        
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
    
print("Model loaded.")

# We only consider test for baseline evaluation
ling_test = linguistic_dataset["test"]
math_test = meta_math_dataset["test"]


# Decrease ling test to 2 for testing
#ling_test = ling_test.select(range(2))
#math_test = math_test.select(range(2))

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



 