# Script that does finetuning of the Mistral7b model
# train and test using LoRA of different parameters of 'r' and domain datasets
# generates the output in the form of a json file for each parameter and domain

#Imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
import time
import json
import os

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


tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        padding="max_length"
    )
    #result['labels'] = result['input_ids'].clone()
    return result

def create_prompt(sample):
    bos_token, eos_token  = "<s>", "</s>"
    
    full_prompt = bos_token
    full_prompt += "### Query:"
    full_prompt += "\n" + sample["prompt"]
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + sample["response"]
    full_prompt += eos_token

    return tokenize(full_prompt)


def format_data_prompt_response(data):
    dataset = { "train": [], "test": [] }
    train, test = data[0], data[1]
    for key in dataset.keys():

        if key == "train":
            for sample in train:
                    tokenized_sample = create_prompt(sample)
                    # Prepare the dataset with input_ids and labels
                    dataset[key].append({
                        "input_ids": tokenized_sample["input_ids"],
                        "labels": tokenized_sample["input_ids"]  # In causal LM, labels are the same as input_ids
                    })
            else:
                for sample in test:
                    tokenized_sample = create_prompt(sample)
                    # Prepare the dataset with input_ids and labels
                    dataset[key].append({
                        "input_ids": tokenized_sample["input_ids"],
                        "labels": tokenized_sample["input_ids"]  # In causal LM, labels are the same as input_ids
                    })
    return dataset


# LoRA parameters
lora_r = [1, 4, 8, 16, 32]




# Transfer model to device
model = model.to(DEVICE)

for key, value in data_dict.items():
    for r in lora_r:
        print(f"Training on domain {key} with r={r}...")
        # LoRA Configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=r*2, # Proportional to rank
            lora_dropout=0.05, # Based on article
            target_modules=["q_proj", "v_proj"]
        )

        # Load data
        with open(DATASET + value["Input"]["Train"], 'r') as file:
            train_dataset = json.load(file)

        with open(DATASET + value["Input"]["Test"], 'r') as file:
            test_dataset = json.load(file)
        
        dataset = format_data_prompt_response([train_dataset, test_dataset])

        
        print(dataset.keys())
       
        # For test and train datasets
        #for key in dataset.keys():
            # Tokenize the list of prompts
        #    dataset[key] = tokenizer(dataset[key], return_tensors="pt", padding=True, truncation=True)
        
        ## Test that data['train'] has labels and input_ids for each sample
        

            
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)

        # Train
        training_args = TrainingArguments( # TODO SPLITS NOT WORKING
            output_dir=OUTPUT_DIR,
            num_train_epochs=1, # TODO: Should be changed
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            logging_steps=10,
            do_train=True,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=10,
            save_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to="none"
        )
       
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],  # use validation for evaluation
            tokenizer=tokenizer
        )

        # Start timing and training
        start = time.time()
        trainer.train()
        end = time.time()
        print(f"Training time: {end - start}")

        # Test
        # For each of the prompt-response pairs in the test dataset, generate a response
        # Format: { hash(prompt): { "prompt": prompt, "response": response, "actual": actual_response } }
        results = {}
        for i, data in enumerate(test_dataset):
            prompt = data["prompt"]
            actual_response = data["response"]
            model_inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)
            generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
            response = tokenizer.batch_decode(generated_ids)[0]

            results[hash(prompt)] = {
                "prompt": prompt,
                "response": response,
                "actual": actual_response
            }

        # Create the directory if it does not exist
        output_dir = value["Output"] + "/r" + str(r)
        os.makedirs(output_dir, exist_ok=True)

        # Write the results to the JSON file
        with open(os.path.join(output_dir, "results_r" + str(r) + ".json"), 'w') as f:
            json.dump(results, f)

        # Write time in a file.txt
        with open(os.path.join(output_dir, "time_r" + str(r) + ".txt"), 'w') as f:
            f.write(str(end - start))
    

