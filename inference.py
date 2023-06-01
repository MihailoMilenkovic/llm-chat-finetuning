from transformers import AutoTokenizer, BloomModel
import torch

from consts import *

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
model = BloomModel.from_pretrained(PRETRAINED_MODEL)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

print(outputs)