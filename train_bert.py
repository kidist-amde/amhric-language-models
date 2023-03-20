import torch
import IPython
from torch.utils.data import Dataset
from tokenizers.processors import BertProcessing
from transformers import DataCollatorForLanguageModeling 
from transformers import Trainer, TrainingArguments
from pathlib import Path
from transformers import BertConfig, BertForMaskedLM , BertTokenizer
import math
from matplotlib.pyplot import *


class AmharicTextDataset(Dataset):
    def __init__(self):
        tokenizer = BertTokenizer.from_pretrained(
            "./logs/am_bert/am_bert-vocab.txt",max_len = 128
        )
       
        self.examples = []
        src_files  = [Path("dataset/amwiki.txt")]


        for src_file in src_files:
            print("🙃", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            for line in lines:
                 
                self.examples.append( tokenizer(line)["input_ids"][:128])
                
                    
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        ids = [token for token in self.examples[i] if token is not None]
        return torch.tensor(ids)


dataset = AmharicTextDataset()

# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig(vocab_size = dataset.tokenizer.vocab_size + 5,

                           )

# Initializing a model (with random weights) from the bert-base-uncased style configuration
model = BertForMaskedLM(configuration)

print(model.num_parameters())


data_collator = DataCollatorForLanguageModeling(
    tokenizer=dataset.tokenizer, mlm=True, mlm_probability=0.15
)


training_args = TrainingArguments(
    output_dir="./logs/am_bert",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_gpu_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model("./logs/am_bert")

import matplotlib.pyplot as plt

def plot_dict(data_dict, start_step, step_size, use_title, use_xlabel, use_ylabel, magnify, ax=None):
    """
    Plots a dictionary of data.

    :param data_dict: A dictionary containing the data to plot.
    :param start_step: The starting step for the x-axis.
    :param step_size: The step size for the x-axis.
    :param use_title: The title of the plot.
    :param use_xlabel: The x-axis label.
    :param use_ylabel: The y-axis label.
    :param magnify: The magnification factor for the y-axis.
    :param ax: The axis object to plot the data on.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot([start_step + i * step_size for i in range(len(list(data_dict.values())[0]))],
            list(data_dict.values())[0])
    ax.set_title(use_title)
    ax.set_xlabel(use_xlabel)
    ax.set_ylabel(use_ylabel)
    ax.set_ylim([min(list(data_dict.values())[0]), magnify * max(list(data_dict.values())[0])])


# Keep track of train loss.
loss_history = {'train_loss': []}

# Keep track of train perplexity.
perplexity_history = {'train_perplexity': []}

# Loop through each log history.
for log_history in trainer.state.log_history:

    if 'loss' in log_history.keys():
        # Deal with training loss.
        loss_history['train_loss'].append(log_history['loss'])
        perplexity_history['train_perplexity'].append(math.exp(log_history['loss']))

# Set up a 1x2 subplot grid.
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot Training Loss.
plot_dict(loss_history, start_step=training_args.logging_steps,
          step_size=training_args.logging_steps, use_title='Loss',
          use_xlabel='Train Steps', use_ylabel='Values', magnify=2, ax=axs[0])

# Plot Training Perplexities.
plot_dict(perplexity_history, start_step=training_args.logging_steps,
          step_size=training_args.logging_steps, use_title='Perplexity',
          use_xlabel='Train Steps', use_ylabel='Values', magnify=2, ax=axs[1])

# Set a tight layout to avoid overlapping labels.
fig.tight_layout()

# Show the plot.
plt.show()
