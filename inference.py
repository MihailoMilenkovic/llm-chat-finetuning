from transformers import AutoTokenizer, BloomModel
import torch
import os

from data_preprocessing import append_to_context_below
from consts import *

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
# pretrained_model = BloomModel.from_pretrained(PRETRAINED_MODEL)
finetuning_checkpoint_path=os.path.join(MODEL_DIR, f'ckpt_epoch_{training_params["num_epochs"]}.pt')
finetuning_checkpoint_data=torch.load(finetuning_checkpoint_path)
model=finetuning_checkpoint_data["model"]
model.eval()
def remove_trailing_end(response):
  parts=response.rsplit(END_KEY, 1)
  return parts[0] if len(parts)>1 else response

def generate_response(prompt,context=""):
  input_text=PROMPT_FORMAT_BEFORE_RESPONSE.format(
    context=context,
    instruction=prompt
  )
  context=append_to_context_below(context, "prompter", prompt)
  inputs = tokenizer(input_text, return_tensors="pt")
  
  #using default greedy decoding, should consider switching to sampling later
  outputs = model.generate(**inputs)
  response=tokenizer.batch_decode(outputs)
  print("response before formatting", response)
  response=remove_trailing_end(response)
  print("response after formatting", response)
  
  context=append_to_context_below(context, "assistant", response)
  return response, context