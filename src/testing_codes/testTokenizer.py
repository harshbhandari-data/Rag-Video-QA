from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-m3")

text = "Machine Learning is computer science jahan pe aap statistical techniques use karte ho."


# tokens=tokenizer.tokenize(text)
# print(tokens)
# print(len(tokens))

tokens_id=tokenizer.encode(text,add_special_tokens=False)  # to remove the 2 special tokens
print(tokens_id)
print(len(tokens_id))