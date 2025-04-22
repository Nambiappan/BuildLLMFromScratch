import os
import re
import importlib
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

print("tiktoken version:", importlib.metadata.version("tiktoken"))
print("pytorch version:", torch.__version__)


with open('tokenise_text_ch02/the-verdict.txt',"r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("Total number of tokens:", len(preprocessed))
# print("First 30 tokens:", preprocessed[:30])

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
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
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

#print("Encoded text:", ids)

decoded_text = tokeniser.decode(ids)
#print("Decoded text:", decoded_text)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer, token in enumerate(all_tokens)}

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

tokeniser = SimpleTokeniser(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)

tokeniser.encode(text)
print(tokeniser.encode(text))

tokeniser.decode(tokeniser.encode(text))
print(tokeniser.decode(tokeniser.encode(text)))

tokeniser = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokeniser.encode(text, allowed_special={"<|endoftext|>"})
print("Encoded text:", integers)
decoded_text = tokeniser.decode(integers)
print("Decoded text:", decoded_text)

enc_text = tokeniser.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(f"Context: {context} --> Desired: {desired}")
    print(f"Context: {tokeniser.decode(context)} --> Desired: {tokeniser.decode([desired])}")

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokeniser, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenise the text
        token_ids = tokeniser.encode(txt,)