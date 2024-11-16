import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
import time
import os

# Comment out in HPC
#token = "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"

# Used in HPC
userToken = os.getenv("HUGGINGFACE_HUB_TOKEN")
# userToken = "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"

if userToken:
    print("Token found.")
    login(token=userToken)


print("Test script started...")
# Constants
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "./mistral_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"... on device {DEVICE}")


# LoRA Configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"Download model ...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"... model downloaded")
print(f"Transfer model to device ...")
model = model.to(DEVICE)
print(f"... model transferred to device")

# generate some prompts
prompt = "My favourite condiment is"
model_inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
print(f"Prompt: {prompt}")
print("Generated response: ")
print(tokenizer.batch_decode(generated_ids)[0])



# # Apply LoRA
# model = get_peft_model(model, lora_config)

# # Load Dataset
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)
# tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# # Prepare DataLoader
# train_loader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

# # Define Training Arguments
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=4,
#     num_train_epochs=1,
#     learning_rate=1e-4,
#     logging_steps=10,
#     save_steps=100,
#     evaluation_strategy="no",
#     fp16=torch.cuda.is_available()
# )

# # Define Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset
# )

# # Fine-tune the model
# print("Starting fine-tuning...")
# trainer.train()
# print("Fine-tuning completed.")

# # Save the fine-tuned model
# model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

# # Benchmark Function
# def benchmark_model(model, tokenizer, prompt, num_tokens=50):
#     model.eval()
#     inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
#     start_time = time.time()

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=num_tokens,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95
#         )

#     end_time = time.time()
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#     elapsed_time = end_time - start_time
#     print(f"\nBenchmark Result:")
#     print(f"Generated Text: {generated_text}")
#     print(f"Time Taken: {elapsed_time:.2f} seconds for {num_tokens} tokens.")
#     return elapsed_time

# # Benchmark the fine-tuned model
# benchmark_prompt = "Once upon a time in a distant land"
# print("\nBenchmarking the fine-tuned model...")
# benchmark_time = benchmark_model(model, tokenizer, benchmark_prompt)

# # Output benchmarking time
# print(f"\nBenchmarking time: {benchmark_time:.2f} seconds")
