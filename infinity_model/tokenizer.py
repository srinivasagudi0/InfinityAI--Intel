import os

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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"<PAD>\t0\n")
            f.write(f"<UNK>\t1\n")
            # save regular charecters sorted by ID, skip the special tokens
            sorted_chars = sorted([(char, id) for char, id in self.char_to_id.items() if char not in ["<PAD>", "<UNK>"]], key=lambda x: x[1])
            for char, id in sorted_chars:
                f.write(f"{repr(char)}\t{id}\n")
        print(f"Vocabulary saved to {file_path}.")

    def load_vocab(self, file_path):
    # load previosly saved vocab from file
        self.char_to_id = {}
        self.id_to_char = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: # skip empty lines as it can cause error when spliting
                    continue
                parts = line.split('\t')
                if len(parts) != 2:  # skip malformed lines
                    continue
                char_repr, id_str = parts

                if char_repr in ("<PAD>", "<UNK>"):
                    char = char_repr
                else:
                    char = eval(char_repr)  # convert repr back to char

                id = int(id_str)
                self.char_to_id[char] = id
                self.id_to_char[id] = char
        self.vocab_size = len(self.char_to_id)
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
    vocab_path = "data/vocab.txt"
    tokenizer.save_vocab(vocab_path)

    # test loading it back
    tokenizer2 = Tokenizer()
    tokenizer2.load_vocab(vocab_path)

    reloaded_encoded = tokenizer2.encode(test_sentence)
    print(f"Reloaded Encoded: {reloaded_encoded}")
    print("Reloaded match:", encoded == reloaded_encoded) # expected: True