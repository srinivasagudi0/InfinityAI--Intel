# the comments are wirrtten so that i can expand this file in future and make it easy for me and contributors, plase understand that this is not AI generated, Thanks.
import os
import torch
from tokenizer import Tokenizer
from word_tokenizer import WordTokenizer
# CONFIG
BLOCK_SIZE = 32 # how many tokens in each training example
BATCH_SIZE = 4 # how many training examples to grab once, more than 10 will cause my comp to crash
TRAIN_SPLIT = 0.9 # 90% train and 10% val

# dataset class
class InfinityDataset:

    def __init__(self, corpus_path, vocab_path, block_size = BLOCK_SIZE, train_split = TRAIN_SPLIT):
        self.block_size = block_size
        self.train_split = train_split
 
        self.tokenizer = WordTokenizer() # create a new empty tokenizer
    

        if os.path.exists(vocab_path):
            self.tokenizer.load_vocab(vocab_path) # if vocab file exists, load it
        else:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                text = f.read() 
            self.tokenizer.build_vocab(text)
            self.tokenizer.save_vocab(vocab_path)
            # build vocab from the corpus text and save it, if the vocab file doesn't exist.


        with open(corpus_path, 'r', encoding='utf-8') as f: 
            text = f.read()

        all_ids = self.tokenizer.encode(text) # encode the whole corpus text into a list of token ids

        print(f"Total tokens in corpus: {len(all_ids)}") # should be 1035


        data = torch.tensor(all_ids, dtype=torch.long) # convers a plain python list into a Pytorch tensor.
        # A tensor is list of number that lives in a special format Pytorch can do math on extremely fast, for use we have dimenstion of 1, since we are using list of token.

        split_idx = int(len(data) * self.train_split) # multiply total tokens by 0.9, round down to whole number.
        self.train_data = data[:split_idx] # start -> split_idx (not including split_idx) ~ 922
        self.val_data = data[split_idx:] # takes everything from split_idx to the end ~ 1035 - 922 = 113

        print(f"Train tokens: {len(self.train_data)}")
        print(f"Val tokens: {len(self.val_data)}")
        print("Dataset initialized.")

    def get_batch(self, split="train", batch_size = BATCH_SIZE):
        """
        Grab a random batch of input/target pairs.

        split : "train" or "val"
        batch_size : how many sequences to return at once.

        return two tensors, both shape (batch_size, block_size)
        x - input sequences
        y - target sequences (x shifted right by 1)
        """

        data = self.train_data if split == "train" else self.val_data
      
        max_start = len(data) - self.block_size - 1 # -1 because we need to shift the target by 1, so we need to make sure we have enough tokens for both input and target sequences.

        if max_start <= 0:
            raise ValueError(
                f"Corpus too small for block size {self.block_size}. "
                f"Add more text to the corpus or reduce the block size."
            )
        
        starts = torch.randint(0,max_start, (batch_size,))# works same as random.randint

        x = torch.stack([data[s : s + self.block_size] for s in starts])
        y = torch.stack([data[s + 1 : s + 1 + self.block_size] for s in starts])

        return x,y 
    
    def vocab_size(self):
        return self.tokenizer.vocab_size

if __name__ == "__main__":

    corpus_path = "data/tiny_corpus.txt"
    vocab_path = "data/vocab.json"

    dataset = InfinityDataset(corpus_path, vocab_path, block_size=BLOCK_SIZE)

    print()
    print("="*50)
    print("Testing get_batch() - train split")
    print("="*50)

    x,y = dataset.get_batch("train")


    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print()

    print("First example decoded: ")
    print(f" input tokens : {x[0].tolist()}")
    print(f" target tokens : {y[0].tolist()}")

    print()
    print(f"input text : {repr(dataset.tokenizer.decode(x[0].tolist()))}")
    print(f"target text : {repr(dataset.tokenizer.decode(y[0].tolist()))}")
    print()

    print("Shift check")
    print(f" input [0:5]: {x[0][:5].tolist()}")
    print(f" target [0:5]: {y[0][:5].tolist()}")
    print(f" input text [1:6]: {repr(dataset.tokenizer.decode(x[0][1:6].tolist()))}")
    print(f" target text input shift by 1: {x[0][1:].tolist() == y[0][:-1].tolist()}") # should be True for all tokens except the last one

    print()
    print("="*50)
    print("Testing get_batch() - val split")
    print("="*50)


    x_val,y_val = dataset.get_batch("val")

    print(f"Input shape: {x_val.shape}")
    print(f"Target shape: {y_val.shape}")
    print(f"Val input text :  {repr(dataset.tokenizer.decode(x_val[0].tolist()))}")

    # end This one should give me mmore data 
    