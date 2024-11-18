# Script that does finetuning of the Mistral7b model
# train and test using LoRA of different parameters of 'r' and domain datasets
# generates the output in the form of a json file for each parameter and domain

#Imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
import time
import json
import os
#import evaluate

# Constants
DATASET = "Dataset/"
data_dict = {
    "Math": {
        "Input": {
            "Train": "Math/Train/MetaMathQA-395K-train.json",
            "Test": "Math/Test/MetaMathQA-395K-test.json"
        },
        "Output": "LoRA/Math/"
    },
    "Linguistic": {
        "Input": {
            "Train": "Linguistic/Test/DopeOrNope-test.json",
            "Test": "Linguistic/Test/DopeOrNope-test.json"
        },
        "Output": "LoRA/Linguistic/"
    }
}
def create_prompt(sample):
    bos_token, eos_token  = "<s>", "</s>"
    
    full_prompt = bos_token
    full_prompt += "### Query:"
    full_prompt += "\n" + sample["prompt"]
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + sample["response"]
    full_prompt += eos_token

    return full_prompt

def format_data_prompt_response(data):
    # Define the formatted dataset structure
    train_data = []
    test_data = []
    train, test = data[0], data[1]

    # Tokenize training data
    for sample in train:
        tokenized_sample = create_prompt(sample)
        train_data.append({
            "input_ids": tokenized_sample["input_ids"],
            "labels": tokenized_sample["input_ids"],
            "attention_mask": tokenized_sample["attention_mask"]
        })

    # Tokenize testing data
    for sample in test:
        tokenized_sample = create_prompt(sample)
        test_data.append({
            "input_ids": tokenized_sample["input_ids"],
            "labels": tokenized_sample["input_ids"],
            "attention_mask": tokenized_sample["attention_mask"]
        })

    # Convert lists to Dataset objects
    train_dataset = Dataset.from_dict({key: [d[key] for d in train_data] for key in train_data[0]})
    test_dataset = Dataset.from_dict({key: [d[key] for d in test_data] for key in test_data[0]})

    return {"train": train_dataset, "test": test_dataset} 



# LoRA parameters
lora_r = [1, 4, 8, 16, 32]


# Used in HPC
userToken = os.getenv("HUGGINGFACE_HUB_TOKEN")

if userToken:
    print("Token found.")
    login(token=userToken)

print("Script started...")

# Constants
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "./mistral_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELPATH = "ModelMistral"



for key, value in data_dict.items():
    for r in lora_r:
        print(f"Training on domain {key} with r={r}...")
    

        # Load data
        with open(DATASET + value["Input"]["Train"], 'r') as file:
            train_dataset = json.load(file)

        with open(DATASET + value["Input"]["Test"], 'r') as file:
            test_dataset = json.load(file)
        
        dataset = format_data_prompt_response([train_dataset, test_dataset])
        
        print(dataset)
        print(dataset.keys())
        print(dataset["train"].keys())
        print(dataset["train"]["input_ids"])
        print(dataset["train"]["labels"])
        print(dataset["train"]["attention_mask"])
        
        