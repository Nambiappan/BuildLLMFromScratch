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


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokeniser, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenise the text
        token_ids = tokeniser.encode(txt, allowed_special ={"<|endoftext|>"})
        print("Total number of tokens:", len(token_ids))
        print(f"max_length --> {max_length}")
        print(f"stride --> {stride}")
        # Check that the number of tokenised inputs is greater than max_length
        assert len(token_ids) > max_length, "Number of tokenised inputs must be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    
    # initialise the tokeniser
    tokeniser = tiktoken.get_encoding("gpt2")

    # Create the dataset
    dataset = GPTDatasetV1(txt, tokeniser, max_length, stride)

    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=4,
    stride=1,
    shuffle=False,
    drop_last=True,
    num_workers=0
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
