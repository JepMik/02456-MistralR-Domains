# Script that does finetuning of the Mistral7b model
# train and test using LoRA of different parameters of 'r' and domain datasets
# generates the output in the form of a json file for each parameter and domain

#Imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
import time
import json
import os

# Constants
DATASET = "Dataset/"
data_dict = {
        "Math": 
            { "Input": {
                "Train": "Math/Train/MetaMathQA-395K-train.json",
                "Test": "Math/Test/MetaMathQA-395K-test.json"
            },
            "Output": "LoRA/Math/"
            },
        
        "Linguistic": 
            { "Input": {
                "Train": "Linguistic\Test\DopeOrNope-test.json",
                "Test": "Linguistic\Test\DopeOrNope-test.json"
            },
            "Output": "LoRA/Linguistic/"
            }
        }  

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
            lora_alpha=32,
            lora_dropout=0.05, # Based on article
            target_modules=["q_proj", "v_proj"]
        )

        # Load data
        train_dataset = load_dataset("json", data_files=DATASET + value["Input"]["Train"])
        test_dataset = load_dataset("json", data_files=DATASET + value["Input"]["Test"])

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        # Train
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1, # TODO: Should be changed
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            logging_steps=10,
            do_train=True,
            do_eval=True,
            evaluation_strategy="steps",
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
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
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
            

