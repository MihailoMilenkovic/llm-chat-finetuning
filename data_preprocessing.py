from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

from consts import *

def append_to_context_above(prev_context,role,message):
  new_context="""
    {new_key}
    {new_utterance}
    {prev_context}
  """.format(
      new_key="{new_key}",
      new_utterance="{new_utterance}",
      prev_context=prev_context)
  if role=="assistant":
    new_context=new_context.format(
      new_key=RESPONSE_KEY,
      new_utterance=message
    )
  else:
    new_context=new_context.format(
      new_key=INSTRUCTION_KEY,
      new_utterance=message
    )
  return new_context

def append_to_context_below(prev_context,role,message):
  new_context="""
    {prev_context}
    {new_key}
    {new_utterance}
  """.format(
      new_key="{new_key}",
      new_utterance="{new_utterance}",
      prev_context=prev_context)
  if role=="assistant":
    new_context=new_context.format(
      new_key=RESPONSE_KEY,
      new_utterance=message
    )
  else:
    new_context=new_context.format(
      new_key=INSTRUCTION_KEY,
      new_utterance=message
    )
  return new_context

def filter_top_replies(dataset):
  to_keep=[False for _ in range(len(dataset.index))]
  grouped_messages = dataset.groupby('parent_id')['message_id'].apply(lambda x: x.index.tolist()).reset_index()
  dataset = dataset.merge(grouped_messages, left_on='message_id', right_on='parent_id', suffixes=('', '_grouped'),how="left")
  dataset = dataset.rename(columns={"message_id_grouped":"all_replies"})
  dataset['all_replies']=dataset['all_replies'].fillna("").apply(list)
  def dfs(index):
    to_keep[index]=True
    curr_col=dataset.loc[index]
    #faster version is to have a field for this processed above
    # curr_children=dataset.loc[dataset["parent_id"]==curr_col["message_id"]]
    curr_children=dataset.iloc[curr_col["all_replies"]]
    curr_len=len(curr_children.index)
    if curr_len==0:
      return
    top_reply_index=curr_children.index[0]
    top_reply=curr_children.iloc[0]
    for index,row in curr_children.iterrows():
      if row["rank"]<top_reply["rank"]:
        top_reply=row
        top_reply_index=index
    dfs(top_reply_index)

  for index, row in dataset.iterrows():
    to_keep[index]=False
  for index, row in dataset.iterrows():
    if row["parent_id"]==None:
      dfs(index)
  
  dataset=dataset[to_keep]
  dataset=dataset.loc[~((dataset['role'] == 'prompter') & (dataset['all_replies'].apply(len) == 0))]
  print("data length after filtering",len(dataset.index))
  return dataset


def get_conversation_chains(dataset):
  def get_parent_row(row):
    return dataset[dataset["message_id"]==row["parent_id"]].iloc[0]
  assistant_replies=dataset[dataset["role"]=="assistant"]
  conversation_chains=[]
  for index,reply_row in assistant_replies.iterrows():
    #use only final replies to get full conversation
    if len(reply_row["all_replies"])>0:
      continue
    #we split the conversation up into 3 parts:
    #context: all previous questions and answers
    #instruction: previous user prompt
    #response: the expected assistant response
    response=reply_row["text"]
    instruction_row=get_parent_row(reply_row)
    instruction=instruction_row["text"]
    context=""
    curr_row=instruction_row
    while curr_row["parent_id"]!=None:
      curr_row=get_parent_row(curr_row)
      append_to_context_above(context,curr_row["role"],curr_row["text"])
      curr_row=get_parent_row(curr_row)
      append_to_context_above(context,curr_row["role"],curr_row["text"])
    conversation_chain=PROMPT_FORMAT.format(
        context=context,
        instruction=instruction,
        response=response,
      )
    conversation_chains.append(conversation_chain)
  return conversation_chains


def get_tokenized_dataset(conversation_chains,tokenizer):
  conversation_dataset=[{"text":conversation} for conversation in conversation_chains]
  tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
  def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
  #TODO: check if some sort of end of turn tokens should be added after assistant response or something similar 
  tokenized_dataset = conversation_dataset.map(tokenize_function, batched=True)
  return tokenized_dataset

def preprocess_dataset(dataset):
  dataset=filter_top_replies(dataset)
  dataset=get_conversation_chains(dataset)
  dataset=get_tokenized_dataset(dataset)
  return dataset

def get_preprocessed_dataset(split="train"):
  if not split in ["train","val"]:
    print("dataset only has 'train' and 'val' data")
    exit(1)
  dataset = load_dataset(DEFAULT_TRAINING_DATASET)
  dataset_split = dataset[split].to_pandas()
  processed_dataset=preprocess_dataset(dataset_split)
  return processed_dataset
