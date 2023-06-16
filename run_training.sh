#!/bin/bash
#SBATCH --job-name=llm-finetuning-test
#SBATCH --ntasks=1
#SBATCH --nodelist=n17
#SBATCH --partition=cuda
#SBATCH --time=24:00:00

export CUDA_VISIBLE_DEVICES=0
python trainer.py