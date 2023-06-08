from typing import List,Tuple
import torch
import os
from data_preprocessing import append_to_context_below
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from consts import *

def load_model_tokenizer_for_generate(
    pretrained_model_name_or_path: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads the model and tokenizer so that it can be used for generating responses.

    Args:
        pretrained_model_name_or_path (str): name or path for model

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    return model, tokenizer

model,tokenizer=load_model_tokenizer_for_generate(DEFAULT_MODEL_PATH)

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