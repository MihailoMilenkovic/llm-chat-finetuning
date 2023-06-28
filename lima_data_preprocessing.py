import json
import pandas as pd
from datasets import Dataset,load_dataset

PROMPTER = "human"
ASSISTANT = "gpt"
ROLE_KEY = "from"
TEXT_KEY = "value"
ID_KEY = "id"
CONVERSATION_KEY = "conversations"
LIMA_DATASET_NAME = "GAIR/lima"
#note: need to log in through huggingface cli in order to load this dataset!
#!huggingface-cli login #followed by entering huggingface API key
def load_lima_dataset_fastchat_format(dataset_name=LIMA_DATASET_NAME, split="train"):
  convs=load_dataset(dataset_name)[split]["conversations"]
  data=[]
  for index,conversation in zip(range(len(convs)),convs):
    curr_conv={}
    role=PROMPTER
    curr_conv[ID_KEY]=f"identity_{index}"
    curr_conv["conversations"]=[]
    for utterance in conversation:
      curr_conv["conversations"].append({
          ROLE_KEY:role,
          TEXT_KEY:utterance
      })
      if role==PROMPTER:
        role=ASSISTANT
      else:
        role=PROMPTER
    data.append(curr_conv)
  return data

def lima_to_fastchat_format(dataset_name:str=LIMA_DATASET_NAME):
    dataset_name=LIMA_DATASET_NAME
    base_name=dataset_name.rsplit("/")[-1]
    print("base",base_name)
    splits=["train","test"]
    for split in splits:
      dataset=load_lima_dataset_fastchat_format(split=split)
      json_path=f"fastchat_{base_name}_{split}.json"
      with open(json_path,"w") as file:
        json.dump(dataset,file)
        print("saved json to ",json_path)

if __name__=="__main__":
  lima_to_fastchat_format()