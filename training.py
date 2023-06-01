from transformers import AutoTokenizer, BloomModel
from datasets import load_dataset
import torch

from consts import *

# tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
# model = BloomModel.from_pretrained(PRETRAINED_MODEL)
dataset = load_dataset(DEFAULT_TRAINING_DATASET)
train_dataset = dataset['train']
val = dataset['validation'] 
print(train_dataset[0].keys())
#find index of "### Response: 
for i in range(len(train_dataset)):
  curr_inst=train_dataset[i]['labels']
  quality_idx=curr_inst["name"].index("quality")
  train_dataset[i]["quality"]=curr_inst["value"][quality_idx]
  print(train_dataset[i])