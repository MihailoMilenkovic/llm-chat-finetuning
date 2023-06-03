from transformers import AutoTokenizer, BloomModel
import torch
import os

from data_preprocessing import append_to_context_below
from consts import *

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
# pretrained_model = BloomModel.from_pretrained(PRETRAINED_MODEL)
finetuning_checkpoint_path=os.path.join(MODEL_DIR, f'ckpt_epoch_15.pt')
finetuning_checkpoint_data=torch.load(finetuning_checkpoint_path)
model=finetuning_checkpoint_data["model"]
model.eval()

def generate_response(prompt,context=""):
  input_text=PROMPT_FORMAT_BEFORE_RESPONSE.format(
    context=context,
    instruction=prompt
  )
  context=append_to_context_below(context, "prompter", prompt)
  context=append_to_context_below(context, "assistant", prompt)
  inputs = tokenizer(input_text, return_tensors="pt")
  #using default greedy decoding, should consider switching to sampling later
  outputs = model.generate(**inputs)
  response=tokenizer.batch_decode(outputs)
  return response, context