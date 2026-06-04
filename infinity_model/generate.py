# wait

import torch
from tokenizer import Tokenizer
from model import InfinityTransformer, VOCAB_SIZE, BLOCK_SIZE

# Config
CHECKPOINT_PATH = "checkpoints/infinity-0.pt"
VOCAB_PATH = "data/vocab.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load everything

def load_model():
    # Load tkenizer
    tokenizer = Tokenizer()
    tokenizer.load_vocab(VOCAB_PATH) # 
    
    # Build model architecture
    model = InfinityTransformer(vocab_size=VOCAB_SIZE).to(DEVICE) # creats a fresh model and moves it to CPU

    # load saved weights into model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE) 
    model.load_state_dict(checkpoint['model_state'])
    # restores the model to the exact state it was when saved# restores the model to the exact state it was when saved

    step = checkpoint.get("step", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"Checkpoint loaded - step {step}, val loss {val_loss:.4f}")

    model.eval() # turn off dropout for generation

    return model, tokenizer

# Generate

def generate(model, tokenizer, prompt, max_new_tokens = 100, temperature = 0.8):
    """
    Generate text starting from a prompt.

    prompt: satrting text
    max_new_tokens: how many new charecters to generate
    temperature: controls randomness
                0.2 = safe, repetitive, predictable
                0.8 = balanced
                1.5 = wild, creative, risk taking
    """

    ids = tokenizer.encode(prompt) # the prompt is th 'x'

    # convert to tensor
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE) # shape (1, len(ids)) and 1 is the barch size

    generated = []

    with torch.no_grad(): # no need to track gradients for generation
        for i in range(max_new_tokens):

            x_input = x[:, -BLOCK_SIZE:] # crop to the BLOCK SIZE if the sequenceis getting too long
            
            #forward pass - gert logits for every position
            logits = model(x_input) # (1, T, vocab_size)
            
            # We only care about last position - thats the next token prediction

            logits = logits[:, -1, :] / temperature # (1, vocab_size) and we divide by temperature to control randomness

            # apply temp
            logits = logits / temperature

            # softmax -> probabilities
            probs = torch.softmax(logits, dim=-1) # (1, vocab_size)

            # sample - pick a token based on the probabilities
            # this is not just picking the highest score every time
            # It samples randomly weighted by probability

            next_id = torch.multinomial(probs, num_samples=1).item() # get the index of the next token

            # Append to the sequence and generated list
            x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)], dim=1) # (1, T+1)


            generated.append(next_id)

    return tokenizer.decode(generated)

# Test
if __name__ == "__main__":
    print("="*50)
    print("Infinity0- Text Generation")
    print("="*50)

    model, tokenizer = load_model()

    test = [
        ("The ", 100, 0.5),
        ("The cat ", 80, 0.8),
        ("Water ", 80, 0.8),
        ("The ", 100, 1.2),
        ("Who are you? ", 100, 1.5), # this one is wild question, the model has never seen it before, so it will have to be creative in its response
    ]

    for prompt, tokens, temp in test:
        print(f"Prompt: {repr(prompt)}")
        print(f"Temperature: {temp}")

        output = generate(model, tokenizer, prompt, max_new_tokens=tokens, temperature=temp)
        print(f"Output: {repr(prompt +output)}")
        print()
        print("-"*50)

