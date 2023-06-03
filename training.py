from transformers import BloomModel
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
import os

from data_preprocessing import get_preprocessed_dataset
from consts import *

if __name__=="__main__":
  #using same training setup as described in LIMA paper (https://arxiv.org/pdf/2305.11206.pdf)
  num_epochs=15
  start_learning_rate=1e-5
  end_learning_rate=1e-6
  beta1=0.9
  beta2=0.95
  weight_decay=0.1
  batch_size=64 #using smaller model and larger batch size

  torch.manual_seed(DEFAULT_SEED)
  model = BloomModel.from_pretrained(PRETRAINED_MODEL)
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.to(device)

  train_dataset = get_preprocessed_dataset("train")
  eval_dataset = get_preprocessed_dataset("val")
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
  eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
  out_dir="saved_models"

  num_training_steps = num_epochs * len(train_dataloader)
  optimizer = AdamW(model.parameters(),lr=start_learning_rate,betas=(beta1,beta2),weight_decay=0.1)
  lr_scheduler = LinearLR(optimizer, start_factor=1,end_factor=end_learning_rate/start_learning_rate, total_iters=num_training_steps)
  progress_bar = tqdm(range(num_training_steps))
  accuracy_metric = evaluate.load("accuracy")
  for epoch_num in range(num_epochs):
    model.train()
    for batch in train_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()

      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      progress_bar.update(1)

    model.eval()
    for batch in eval_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
          outputs = model(**batch)

      logits = outputs.logits
      predictions = torch.argmax(logits, dim=-1)
      #TODO: check how to format this
      accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])

    val_accuracy=accuracy_metric.compute()
    #checkpoint model after every 5 epochs

    if epoch_num%5==0:
      checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'model_args': model_args,
        'epoch_num': epoch_num,
        'val_accuracy': val_accuracy,
      }
      print(f"saving checkpoint to {out_dir}")
      torch.save(checkpoint, os.path.join(out_dir, f'ckpt_epoch_{epoch_num}.pt'))
    
  