#!/bin/bash
#SBATCH --job-name=llm-finetuning-test
#SBATCH --ntasks=1
#SBATCH --nodelist=n01
#SBATCH --time=1:00:00
python trainer.py --epochs=1