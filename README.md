# **02456-MistralR-Domains**  
> Investigating cross-domain intelligence of LLM (Mistral7b) by LoRA finetuning on Linguistic and Mathematics datasets with different versions of rank r=[1, 4, 8 ,16]

## **Authors**
- Jeppe Mikkelsen (s204708)
- Nina Peuker (s204669)
- Mario Medoni (s204684)

## **Jupyter Notebook**
As specified in the assignment instructions, the Git repository is required to include a Jupyter notebook capable of reproducing the results. However, this is **not** feasible in our case due to the size of the Mistral 7B model, which consists of 7 billion parameters and requires 27.8 GB of memory. Consequently, most of our work was conducted on the DTU HPC cluster, as it was the only environment capable of meeting these memory requirements.

We also attempted to include the actual LoRA adapters in the repository, but GitHub's file size limitations prevented us from doing so.

For further inquiries or access to the necessary resources, please contact the repository owner

## **Table of Contents**   
1. [Project Structure](#project-structure)  
2. [Datasets](#datasets)  
3. [Results](#results)
4. [Plots](#plots) 
5.  [License](#license)  

## **Project Structure**  
```
02456-MistralR-Domains/  
|
├──|asserts/                   # Plots of all results gathered  
├──|Baseline/                  # Baseline processed datasets  
|  |-- Processed_datasets/     # Contains the train, val and test splits of both datasets in [0.8,0.1,0.1]
|  |-- Results/                # The LLM generated baseline results for each test set and the overall baseline evaluation
|  |-- *                       # Scripts used for datasplits, generating responses, and evaluating and job-scripts for HPC
├──| Evaluation/               # Evaluation results and scripts  
|  |-- Results/                # Contains all results e.g. Generated responses and scores based on datasets and different versions of R 
|  |-- resGen.py               # Prompting ChatGPT 04-mini for comparision
|  |-- rEval.py                # Generating responses for different versions of r and Cross Domain
|--| LoRA/                     # Training scripts
|  |--| tokenized_datasets/    # Contains the pre tokenized dataset for pipeline speedup of training aspect
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation  
```  

## **Datasets**  
### Sources  
- **Math Dataset**: `meta-math/MetaMathQA`(https://huggingface.co/datasets/meta-math/MetaMathQA)
- **Linguistic Dataset**: `DopeorNope/Linguistic_Calibration_Llama3.1`(https://huggingface.co/datasets/DopeorNope/Linguistic_Calibration_Llama3.1)

### Preprocessing  

The datasets were preprocessed into an `Instruction --> Answer` format, leveraging prompt engineering techniques such as the `[INST]` and `[/INST]` markers to structure the input-output pairs effectively. This ensured that the model could understand and align with the intended instruction-based training paradigm.  

For the training of our LoRA adapters, the datasets were pre-tokenized to optimize the training pipeline by reducing redundant computation. This preprocessing step allowed the pipeline to focus on specific tasks during training, improving efficiency and ensuring consistent tokenization across all samples.  

## **Results**  
### Comparison  
Include comparison across different LoRA versions with rank (`r = 1, 4, 8, 16`).

---
We evaluated the generated responses against ground truth using ChatGPT 04-Mini. The evaluation focused on two key domains:  

1. **Mathematical Tasks**  
   - Scored based on metrics such as **problem clarity**, **logical structure**, and **arithmetic complexity**.  
   - Correct answers contributed 40% of the total score, with the remaining 60% distributed equally across other metrics.  

2. **Linguistic Tasks**  
   - Scored on metrics such as **lexical diversity**, **syntactic complexity**, and **readability**.  
   - Ground truth responses were treated as the baseline with a score of 100.

### **Evaluation Process**  
- **Prompts**: Responses and ground truths were formatted into task-specific prompts.  
- **Scoring**: ChatGPT API was used to score the responses, ensuring consistency with temperature set to 0.  
- **Results Storage**: Scores were saved in CSV files for each LoRA rank (`R1`, `R4`, `R8`, `R16`).  

The results provide a comprehensive comparison of fine-tuned configurations across tasks, showcasing model performance under varying setups.


## **Plots**

### Method
The method utilised througout the project

![Method](/assets/Method.png)


### Own Domain
Results in Bar-plot in own domain of knowledge with baseline included

![Own Domain](/assets/Own_Domain.png)

### Cross Domain
Results of Cross Domain

![Cross Domain](/assets/Cross_Domain.png)



## **License**  
This project is licensed under the MIT License. See the `LICENSE` file for details.  

---  

Let me know if you'd like me to expand on any specific section or tailor this template further!
