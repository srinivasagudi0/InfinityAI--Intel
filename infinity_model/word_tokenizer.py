# splits text into WORDS instead of charecters

import os
import re
import json
from collections import Counter

class WordTokenizer:

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def tokenize_words(self, text):
        # split text into tokens and words and punctuation are treated as separate tokens
        return re.findall(r"\w+|[^\w\s]", text)
    
    def build_vocab(self, text, max_vocab_size = 8000):
        #build vocabulary from the most commoon words

        words = self.tokenize_words(text)
        counts = Counter(words)

        most_common = counts.most_common(max_vocab_size-2) # reserve 2 for special tokens

        self.word_to_id = {"PAD":0, "UNK":1} # PAD for padding, UNK for unknown words
        self.id_to_word = {0:"PAD", 1:"UNK"}


        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id)
        print(f"Vocab Built."
              f"Size: {self.vocab_size} unique words"
              f"Out of {len(counts)} total words seen.")        

    def encode(self, text):

        words = self.tokenize_words(text)
        return [self.word_to_id.get(w, 1) for w in words]
    
    def decode(self, ids):
        text = ""
        words = [self.id_to_word.get(i, "UNK") for i in ids]
        for i, w  in enumerate(words):
            if i>0 and w not in ".,!?;:()\"'" and words[i-1] not in "(\"":
                text += " "
            text += w
        return text
    
    def save_vocab(self, path):
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        DATA = {
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word
        }

        with open(path, "w") as f:
            json.dump(DATA, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path):
        with open(path, "r") as f:
            DATA = json.load(f)
        
        self.vocab_size = DATA["vocab_size"]
        self.word_to_id = DATA["word_to_id"]
        self.id_to_word = {int(k):v for k,v in DATA["id_to_word"].items()}

        print(f"Vocabulary loaded successfully. Size: {self.vocab_size} unique words")

if __name__ == "__main__":
    corpus_path = "data/tiny_corpus.txt"
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = WordTokenizer()
    tokenizer.build_vocab(text, max_vocab_size=8000)
    print()

    test = "The cat sat on the mat."
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)

    print(f"Original : {test}")
    print(f"Encoded  : {encoded}")
    print(f"Decoded  : {decoded}")
    print()

    tokenizer.save_vocab("data/word_vocab.json")

    tokenizer2 = WordTokenizer()
    tokenizer2.load_vocab("data/word_vocab.json")
    reloaded = tokenizer2.encode(test)
    print("Reload match:", reloaded == encoded)

