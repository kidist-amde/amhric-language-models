from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./logs/am_bert/am_bert-vocab.json",
    "./logs/am_bert/am_bert-merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print(
    tokenizer.encode("በኢትዮጵያ ዘመን አቆጣጠር")
)

tokens =  tokenizer.encode("በኢትዮጵያ ዘመን አቆጣጠር")
print(tokens.ids)
print(tokenizer.decode(tokens.ids))

