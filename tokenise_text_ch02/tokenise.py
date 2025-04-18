import os
import re

with open('tokenise_text_ch02/the-verdict.txt',"r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("Total number of tokens:", len(preprocessed))
print("First 30 tokens:", preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print("Vocabulary size:", vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(f"{i}: {item}")
    if i > 50:
        break

class SimpleTokeniser:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokeniser = SimpleTokeniser(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""

ids = tokeniser.encode(text)

print("Encoded text:", ids)

decoded_text = tokeniser.decode(ids)
print("Decoded text:", decoded_text)

tokeniser = SimpleTokeniser(vocab)
text = "Hello, do you like tea. Is this-- a test?"

tokeniser.encode(text)

all_tokens = sorted(list(set(preprocessed)))

