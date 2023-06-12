#!/bin/bash
#SBATCH --job-name=llm-finetuning-test
#SBATCH --ntasks=1
#SBATCH --nodelist=n17
#SBATCH --partition=cuda
#SBATCH --time=1:00:00
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
python trainer.py --epochs=1