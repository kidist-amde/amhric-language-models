#! pip install tokenizers

from pathlib import Path
from tokenizers import  BertWordPieceTokenizer



paths = ["dataset/amwiki.txt"]

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("logs/am_bert", "am_bert")

