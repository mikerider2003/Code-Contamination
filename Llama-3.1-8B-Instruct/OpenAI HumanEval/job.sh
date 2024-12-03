#!/bin/bash
#SBATCH --job-name=OpenAI_HumanEval_test         
#SBATCH --output=OpenAI_HumanEval_test.out         
#SBATCH --error=OpenAI_HumanEval_test.err          
#SBATCH --time=04:00:00             
#SBATCH --partition=gpu
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=16     
#SBATCH --mem=90G                     
#SBATCH --gres=gpu:h100:1

module load Anaconda3
module load CUDA/12.1.1
conda init
source ~/.bashrc
conda activate env2
python --version
python run.py
