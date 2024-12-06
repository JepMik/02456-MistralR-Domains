# Imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset, load_from_disk, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from accelerate import Accelerator
import time
import json
import os
import sys
import matplotlib.pyplot as plt

# Constants
MODELPATH = "ModelMistral"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
PROCESSED_DIR = "LoRA/Working/tokenized_datasets"
R = [4]

# Initialize the Accelerator
accelerator = Accelerator()

# Device setup
DEVICE = accelerator.device
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

arg = sys.argv[1]
isMath = arg == "Math"

# Load tokenized datasets
def load_tokenized_datasets(IsMath: bool):
    print(f"{PROCESSED_DIR}/meta_math_tokenized" if IsMath else f"{PROCESSED_DIR}/linguistic_tokenized")
    if os.path.exists(f"{PROCESSED_DIR}/linguistic_tokenized") and os.path.exists(f"{PROCESSED_DIR}/meta_math_tokenized"):
        print("Loading processed datasets from local disk...")
        if isMath:
            return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math_tokenized")
        else:
            return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/linguistic_tokenized")
    else:
        raise Exception("Tokenized datasets not found")

def load_model():
    if os.path.exists(MODELPATH) and os.listdir(MODELPATH):
        print(f"Loading model from {MODELPATH}...")
        model = AutoModelForCausalLM.from_pretrained(MODELPATH)
        tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
    else:
        print(f"Downloading model from {MODEL_NAME}...")
        userToken = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if userToken:
            print("Token found.")
            login(token=userToken)

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print(f"Saving model to {MODELPATH}...")
        model.save_pretrained(MODELPATH)
        tokenizer.save_pretrained(MODELPATH)
    return model, tokenizer

print("Loading model and tokenizer...")

model, saved_tokenizer = load_model()
tokenized_data = load_tokenized_datasets(isMath)

#saved_tokenized_data = tokenized_data

print("Starting freezing...")
# Freeze the base model weights
for param in model.parameters():
    param.requires_grad = False

print("Starting fine-tuning...")
for r in R:
    tokenizer = saved_tokenizer
    model = model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Fine-tuning with R={r}...")

    #tokenized_data = saved_tokenized_data
    tokenizer.pad_token = tokenizer.eos_token


    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=r * 2,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj","o_proj","gate_proj"]
    )

    model = get_peft_model(model, lora_config)

    model, tokenized_data["train"], tokenized_data["val"] = accelerator.prepare(
        model, tokenized_data["train"], tokenized_data["val"]
    )

    output_dir = f"FineTuned_{'Math' if isMath else 'Linguistic'}_R{r}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        num_train_epochs=10,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_accumulation_steps=8,
    )

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["val"],
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback],
    )

    print("Starting training...")
    start = time.time()
    trainer.train()
    end = time.time()
    print(f"Training time: {end - start} seconds")

    # Save only the LoRA weights
    lora_output_dir = f"New_test/{output_dir}/lora_weights"
    model.save_pretrained(lora_output_dir)
    print(f"LoRA weights saved to {lora_output_dir}")

  
#FineTuned_Linguistic_R1_Test/loss_plot.png