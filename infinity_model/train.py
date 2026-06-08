# Every Step:
# 1. grab a batch of input/target pairs
# 2. model makes predicrtions (forward pass)
# 3. measure how wrong the predictions are (loss)
# 4. figure out which weights caused the error (backpropagation)
# 5. nudge those weights in the right direction (optimizer step)
# reeat

import os
import torch
from tokenizer import Tokenizer
from dataset import InfinityDataset
from model import InfinityTransformer, VOCAB_SIZE, BLOCK_SIZE
import time

# Config
STEPS = 3000 # how many training step to run 
BATCH_SIZE = 8 # examples per step
LEARNING_RATE  = 1e-3 # how big each weight nudge is (0.001)
EVAL_EVERY = 200 # print loss every 100 steps
PATIENCE  = 5 # stop early if val loss doesn't improve this many evalss


CORPUS_PATH = "data/tiny_corpus.txt"
VOCAB__PATH = "data/vocab.json"

# Use gpu if available
DEVICE ="cuda" if torch.cuda.is_available() else "cpu"


def get_run_dir():
    os.makedirs("runs", exist_ok=True)
    existing = [d for d in os.listdir("runs") if d.startswith("run")]
    num = len(existing) + 1
    path = f"runs/run_{num:03d}"
    os.makedirs(path)
    return path

# Estimate loss
# We check loss on both train and val data every 100 steps and print it out

@torch.no_grad() # don't need to track gradients as it saves memory and computations
def estimate_loss(model, dataset, eval_steps=100): # test the moddel fr 100 batches first
    "Health check!"
    model.eval() # switch to evaluation mode, also making evaluation faster and more stable
    losses = {}
    for split in ['train', 'val']:
        total = 0.0
        for i in range(eval_steps):
            x, y = dataset.get_batch(split, batch_size=BATCH_SIZE) # x = input y = target toekns
            x, y = x.to(DEVICE), y.to(DEVICE) # Move to cpu (in out caese it is not gpu)
            logits = model(x) # predictions for each and every token

            # reshape for cross entropy:
            # logits.view(-1, VOCAB_SIZE): (B, T, vocab_size) -> (B*T, vocab_size)
            # y.view(-1): (B, T) -> (B*T)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                y.view(-1)
            ) # this compares predicted next token, actual next toekn for every position
            total +=loss.item()
            losses[split] = total / eval_steps # store average loss for this split
        model.train() # switch back to training mode, re enabling dropout and other regularization techniques
    return losses # expecting a dict like {"train": 2.5, "val": 2.7}


# Main training loop

def train():

    run_dir = get_run_dir()
    log_path = os.path.join(run_dir, "log.txt")
    ckpt_path = os.path .join(run_dir, "best_checkpoint.pt")
    print(f"Run folder : {run_dir}")

    def log(msg):
        print(msg)
        with open(log_path,"a") as f:
            f.write(msg + "\n")

    log("=="*25)
    log(f"Infinity-0 Training - {run_dir}")
    print(f"Device: {DEVICE}")
    log(f"Steps: {STEPS}")
    log(f"Batch Size: {BATCH_SIZE}")
    log(f"Block Size: {BLOCK_SIZE}")
    log(f"Block Size: {BLOCK_SIZE}")

    # Step 1 - load dataset (tokenizer is loaded insi)
    dataset = InfinityDataset(CORPUS_PATH, VOCAB__PATH)
    log("Dataset loaded.")

    #Step 2 - Create model and move to devce
    model = InfinityTransformer(
        vocab_size=VOCAB_SIZE,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model has {total_params:,} parameters")

    # step 3 - create an optimizer
    # AdamW is like a smarter version of gradeint descent
    # It adjust elaning rate for each weight indinvidually
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    log(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    log(f"Corpus : {dataset.tokenizer.vocab_size:,} vocab chars")


    # step 4 make chekpoints folder if it doent exit
    os.makedirs("checkpoints", exist_ok=True)


    # step 5 - training loop!!!!!!!!!!!!!!!
    best_val_loss = float('inf')
    start_time = time.time()
    no_improve = 0
    model.train() # already in train mode but just to be sure


    for step in range(STEPS):

        # EVRY EVAL_EVERY steps - prints loss

        if step % EVAL_EVERY == 0:
            losses = estimate_loss(model, dataset, eval_steps=100)
            elapsed = time.time() - start_time
            msg = (f"step {step:4d} | "
                   f"train : {losses['train']:.4f} | "
                   f"val : {losses['val']:.4f} | "
                   f"time: {elapsed:.0f}s"
            )

            # save the best model 
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                no_improve = 0
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "vocab_size": VOCAB_SIZE,
                    "run_dir": run_dir,
                }, ckpt_path)

                actual_checkpoint = "checkpoints/infinity-0.pt"
                torch.save(torch.load(ckpt_path), actual_checkpoint)
                msg += " (checkpoint saved)"
            else:
                no_improve +=1
                msg += f" (no improve {no_improve}/{PATIENCE})"
            log(msg)
            if no_improve >= PATIENCE:
                log("Early stopping due to no improvement in validation loss.")
                break



        # get a batch
        x, y = dataset.get_batch("train", batch_size=BATCH_SIZE) # x = input y = target toekns
        x, y = x.to(DEVICE), y.to(DEVICE) # Move to cpu (in out caese it is not gpu)

        # forward pass
        logits = model(x) # predictions for each and every token; shape (B, T, vocab_size)

        # calculate loss
        # cross_entropy expects (N, C) and (N,) so we flatten

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, VOCAB_SIZE), # (B*T , VOCAB_SIZE)
            y.view(-1) # (B*T,)
        )

        # backpropagation
        optimizer.zero_grad() # clear previous gradients
        loss.backward() # compute NEW gradients

        # clip gradiemts - prebemts them from expoding to huge numbers
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # optimizer step - update weights
        optimizer.step() # nudge weights in the right direction

        # final eval
    losses = estimate_loss(model, dataset, eval_steps=100)
    log("==" *25)
    log(f"Done. Best val loss: {best_val_loss:.4f}")
    log(f"Final train loss: {losses['train']:.4f}")
    log(f"Final val loss: {losses['val']:.4f}")
    log(f"Model saved to {ckpt_path}")
    log("==" *25)


if __name__ == "__main__":
    train()


# i ran the train function and it started overfitting (the model memorizeed the text instead of learning patterns). 
# To fix this, I need to increase the size if the tiny coprus

