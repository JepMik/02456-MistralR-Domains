from datasets import load_dataset, DatasetDict
import os

PROCESSED_DIR = "Baseline/processed_datasets"

def create_prompt(dict,flag_data):
    """
    Create a formatted prompt for the model to generate a response.
    """
    if flag_data:
        dict["query"] = "[INST]\n Provide the answer to the following question: \n" + dict["query"] + "\n[/INST]"
    else: 
        dict["paragraph_generation_prompt"] = "[INST]\n" + dict["paragraph_generation_prompt"] + "\n[/INST]"
    return dict

def load_processed_datasets():
    # Check if processed datasets exist locally
    if os.path.exists(f"{PROCESSED_DIR}/linguistic") and os.path.exists(f"{PROCESSED_DIR}/meta_math"):
        print("Loading processed datasets from local disk...")
        # Load the entire DatasetDict (not just individual splits)
        
        meta_math_dataset = DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math")
        linguistic_dataset = DatasetDict.load_from_disk(f"{PROCESSED_DIR}/linguistic")
    return meta_math_dataset, linguistic_dataset
    

math_data, ling_data = load_processed_datasets()
#print(math_data['train'][0])
#print(ling_data['train'][0])


def format_data(dataDict, isMath):
    for key in dataDict:
        dataDict[key] = dataDict[key].map(lambda x: create_prompt(x, isMath))
        
    return dataDict

math_data = format_data(math_data, True)
ling_data = format_data(ling_data, False)
#print("-----------------------------------------------------")

## Save to processed datasets
# Save processed datasets to disk (entire DatasetDict)
if not os.path.exists(f"{PROCESSED_DIR}/linguistic"):
    ling_data.save_to_disk(f"{PROCESSED_DIR}/linguistic")
if not os.path.exists(f"{PROCESSED_DIR}/meta_math"):
    math_data.save_to_disk(f"{PROCESSED_DIR}/meta_math")
print("Datasets saved to disk.")
