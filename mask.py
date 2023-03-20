from transformers import pipeline, BertModel , BertTokenizer



tokenizer = BertTokenizer.from_pretrained("path-to-tokenizer")
model = BertModel.from_pretrained("path-to-model")
unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
unmasker("የእስራኤል ዋና ከተማ [MASK] ነው።")