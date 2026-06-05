# score infinity-0 output quality
# Run after every training session to  decide if model is ready

import torch
from tokenizer import Tokenizer
from model import InfinityTransformer, VOCAB_SIZE, BLOCK_SIZE

CHECKPOINT = "checkpoints/infinity-0.pt"
VOCAB_PATH = "data/vocab.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PROMPTS = [
    ("The ", 80, 0.5),
    ("Water ", 80, 0.7),
    ("The brain ", 80, 0.8),
    ("Learning ", 80, 0.8),
    ("Artificial " 80, 0.8)
]

def load_model():
    tokenizer = Tokenizer()
    tokenizer.load_vocab(VOCAB_PATH)
    model = InfinityTransformer(vocab_size=VOCAB_SIZE).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    val_loss = ckpt.get("val_loss", "?")
    step = ckpt.get("step", "?")
    return model, tokenizer, step, val_loss

def generate(model, tokenizer, prompt, max_new=80, temperature=0.7):
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device = DEVICE).unsqueeze(0) # (1, T)
    out = []
    with torch.no_grad(): # no wasting power on gradients for evaluation
        for i in range(max_new):
            logits = model(x[:, -BLOCK_SIZE:])
            logits = (logits[:, -1, :] / temperature)
            probs = torch.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1)
            out.append(next_id.item())
    return tokenizer.decode(out)

def score_output(text):
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return {"score": 0, "reason": "empty  output"}
    
    # how many tokens are actual english looking words
    real_words = [w for w in words if w.isalpha() and len(w) >= 2]
    word_ratio = len(real_words) / total_words

    unique_ratio = len(set(words)) / total_words

    # how many unkown tokens are there
    unk_count = text.count("<UNK>")
    unk_ratio = unk_count / max(len(text), 1)

    avg_len = sum(len(w) for w in real_words) / max(len(real_words), 1)

    score = 0

    if word_ratio > 0/55: 
        score += 25
    if unique_ratio > 0.30:
        score += 25
    if unk_ratio < 0.02:
        score +=25
    if 3.0 < avg_len < 8.0:
        score += 25

    return {
        "score": score,
        "word_ratio": word_ratio,
        "unique_ratio": unique_ratio,
        "unk_ratio": unk_ratio,
        "avg_word_len": avg_len,
    }

def evaluate():
    print("=" * 50)
    print("Infinity-0 - Quality Eval")
    print("=" * 50)

    model, tokenizer, step, val_loss = load_model()
    print(f"Checkpoint : step {step}, val loss {val_loss:.4f}")
    print()

    total_score = 0

    for prompt, tokens, temp in TEST_PROMPTS:
        output = generate(model, tokenizer, prompt, max_new=tokens, temperature=temp)
        full = prompt + output
        result = score_output(output)

        print(f"Prompt: {repr(prompt)}")
        print(f"Output: {repr(output)}")
        print(f"Score: {result['score']} / 100")
        print(f"Words: {result['word_ratio']}")
        print(f"variety: {result['unique_ratio']}")
        print(f"unk: {result['unk_ratio']}")
        print(f"avg_len: {result['avg_word_len']}")

        print()
        avg =total_score / len(TEST_PROMPTS)
        
        print("==" *55)
        print(f"Average quality score: {avg:.1f} / 100")
        print()

        if avg >= 60:
            print("Pass - Infinity-0 is producing readable text!")
            print("Move on.")
        elif avg >= 40:
            print("Partial - output is still mostly noisy")
            print("Crawl more data and re train.")
        
        else:
            print("Fail")
            print("Crawl more dat,  check hyperparameters, and re train. ")
            
        print("="*55)
        return avg

if __name__ == "__main__":
    evaluate()

# python crawler.py
# python train.py
# python eval.py
