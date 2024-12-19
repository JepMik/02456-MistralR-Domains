#!/bin/sh 

### Queue specification 
#BSUB -q gpua100

### Name of job
#BSUB -J LoRA

### Number of cores
#BSUB -n 1

### Cores must be on same host
#BSUB -R "span[hosts=1]"

###
#BSUB -R "select[gpu80gb]"

### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=64GB]"

### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
###BSUB -M 64

### -- set walltime limit: hh:mm --
#BSUB -W 24:00

### -- set the email address --
#BSUB -u s204708@dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion --
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

nvidia-smi
module load cuda/11.6

cd $BLACKHOLE/02456-MistralR-Domains
source $BLACKHOLE/env
pip install -r requirements.txt
python LoRA/lora.py Math 1

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery