import os
import json

class Tokenizer:
    def __init__(self):
        self.char_to_id = {} # caused mismatch
        self.id_to_char = {}
        self.next_id = 0

    def build_vocab(self, text):
        unique_chars = sorted(set(text))
        
        # Reserve 2  special tokens
        # 0 = <PAD>
        # 1 = <UNK>

        self.char_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_char = {0: "<PAD>", 1: "<UNK>"}

        # assign id to every charecter starting from 2
        for idx, char in enumerate(unique_chars, start=2):
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char
        
        self.vocab_size = len(self.char_to_id)
        print(f"Vocabulary built. Size: {self.vocab_size} characters.")

    def encode(self, text):
        return [self.char_to_id.get(char, self.char_to_id["<UNK>"]) for char in text]

    def decode(self, ids):
        return ''.join([self.id_to_char.get(id, "<UNK>") for id in ids])

    def save_vocab(self, file_path):
        dir_name = os.path.dirname(file_path)

        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "char_to_id": self.char_to_id,
            "id_to_char": self.id_to_char
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary saved to {file_path}.")

    def load_vocab(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()} # convert keys back to int
        
        print(f"Vocabulary loaded from {file_path}. Size: {self.vocab_size} characters.")



if __name__ == "__main__":
    corpus_path = "data/tiny_corpus.txt"

    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = Tokenizer()

    tokenizer.build_vocab(text)
    print()

    #test encoding and decoding
    test_sentence = "The cat sat on the mat."
    encoded = tokenizer.encode(test_sentence)

    print(f"Original: {test_sentence}")
    print(f"Encoded: {encoded}")

    print()

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    print("Match:", test_sentence == decoded)
    print()

    # test with unknown characters
    unknown_test = "Hello 👋"
    encoded_unk = tokenizer.encode(unknown_test)    
    decoded_unk = tokenizer.decode(encoded_unk)

    print(f"Original with unknown: {repr(unknown_test)}")
    print(f"Encoded with unknown: {encoded_unk}")
    print(f"Decoded with unknown: {repr(decoded_unk)}")
    print("Match with unknown:", unknown_test == decoded_unk) #expecterd: False
    print()

    # save the vocab
    vocab_path = "data/vocab.json"
    tokenizer.save_vocab(vocab_path)

    # test loading it back
    tokenizer2 = Tokenizer()
    tokenizer2.load_vocab(vocab_path)

    reloaded_encoded = tokenizer2.encode(test_sentence)
    print(f"Reloaded Encoded: {reloaded_encoded}")
    print("Reloaded match:", encoded == reloaded_encoded) # expected: True