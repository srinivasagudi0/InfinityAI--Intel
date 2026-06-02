from itertools import count

import torch
import torch.nn as nn

 # TODO: build the model, count parameters, and test a forward pass with fake input. The attention head and multi-head attention are implemented, because they will be used to convert the input tokens into a form that the model can understand and use to make predictions

# Config

VOCAB_SIZE = 44 # number of unique characters in the corpus
BLOCK_SIZE = 32 # max sequence length
N_EMBED = 64 # how many numbers represent each token
N_HEADS = 4 # how many attention heads
N_LAYERS = 2 # how many transformer blocks
DROPOUT = 0.1 # randomly zero out 10% of connection during training.


# Attention head
# One attention hea. It lets each token look at previous tokens 
# and decidde which 

# Example: in "THe cat sat", when predicting after "sat",
# the model might focus heavily on "cat" because it is the subject

# what is torch.nn
# torch.nn is a submodule of PyTorch that provides a set of tools and classes for building neural networks. 
# It includes layers, loss functions, optimizers, and other utilities that make it easier to create and train neural network models. 
# The nn module is designed to work seamlessly with PyTorch's tensor operations, allowing for efficient computation and automatic differentiation.

class AttentionHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        # three learned projections - query, key, value
        #  query - "what am I looking for?"
        # key - "what do I contain?"
        # value - "what do I actually give u?"

        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        # These three lines create 3 diff linear layers that tur each token into query, key and value vectors
        # these arre required for attention to computescores and pass info.

        # all three layers are differnet because "every call to nn.linear creates new random wieght"

        # Mask - prevents tokens from looking at future tokens
        # (you can't predict "cat" by cheating and looking at "cat" itself)
        self.register_buffer(
            "mask", # name of the buffer
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)) #basically a buffer, slowly fills up with 1s in the lower triangle and 0s in the upper triangle, so it creates a mask that allows tokens to only attend to previous tokens and not future tokens.
        ) #  This exists so that it prevents model from cheating diurig training by looking at future tokens when predicting the next token.

    def forward(self, x):
        """
        This function
        - Make Q, K, V
        - Computes attention scores
        - Masks future tokens
        - Turns scores into probabilities
        - Returns the weighted sum of the values
        """
        B, T, C = x.shape # batch, time(sequence length), channels (embed size)

        q = self.query(x) # multiplies x by the query weight matrix
        k = self.key(x) # produces key vectors
        v = self.value(x) # produces value vectors

        ## Attention scores - how much does each token care about each other token?
        # Divide by sqrt(head_size) to keep numbers from getting too large
        scores = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)  # This compares every token with every other token 
        # This line also calculates how much each token should pay attention to every other token.
        
        # Aplly mask - set future positions to -infinity so softmax amkes them zero 
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        
        # Siftmax turns scores into probability dsitribution
        weights = torch.softmax(scores, dim=-1) # converts each row into a probability distribution
        weights = torch.nn.functional.dropout(weights, p=DROPOUT, training=self.training) # randomly drops some attention weights during training

        # Weighted sum of values
        out = weights @ v
        return out
    
    # Multi-Head Attention

# Multi-head attention 
class MultiHeadAttention(nn.Module):

    """
    Each head looks at the input differently
    Their outputs are concatenated(meaning we just put them side by side)
    A final linear layer mixes them
    Dropout is applied
    """

    def __init__(self):
        super().__init__()

        head_size = N_EMBED // N_HEADS # split embedding evenly across heads 
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(N_HEADS)])# each head is an independent attention head
        self.project = nn.Linear(N_EMBED, N_EMBED) # after alll heads are combined, this layer mixes them together
        self.dropout = nn.Dropout(DROPOUT) # randomly drop some values to prevent overfitting


    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1) # it runs each head on the same input x and outputs a chink of embedding
        out = self.dropout(self.project(out)) # mixes all head outputs together into one vectoe and randomly drops value
        return out
    

# Feed Forward 

#After attention, each token thinks independently
# Two linear layers with ReLU in between
# This is where the model stores most of its "knowledge"

class FeedForward(nn.Module):
    "It transforms each token's embedding individually to help the model learn complex patterns"
    def __init__(self):
        super().__init__()
        # nn.Sequantial is used to stack multiple layers together
        # build a small neural network 
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED), # expands the embedding -making it 4x bugger/ gives model more thinking space
            nn.ReLU(), # non -linearity /. making the model able to learn complex patterns
            nn.Linear(4 * N_EMBED, N_EMBED), # balances the embedding -making it the same size as the input
            nn.Dropout(DROPOUT) # random % drop
        )
    
    def forward(self, x):
        return self.net(x) # simply runs x through the stakced layers above
    
    
# transformer block

class TransformerBlock(nn.Module):
    """
    Attention mixes tokens.
    FeedForward transforms tokens
    LayerNorm stabilizes
    Residuals keep info flowing
    """

    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention() # lets token look at each other
        self.feed_forward = FeedForward() # small neural network applied to each token
        self.norm1 = nn.LayerNorm(N_EMBED) # stabilize
        self.norm2 = nn.LayerNorm(N_EMBED) # !!

    def forward(self, x):
        x = x + self.attention(self.norm1(x)) # normalizes the input, runs input and addds the result back to the original input
        x = x + self.feed_forward(self.norm2(x)) #normalizes again, runs feed foward netwoek adn adds the result back

        return x
    
class InfinityTransformer(nn.Module):

    def __init__(self, vocab_size = VOCAB_SIZE):
        super().__init__()

        # Token embedding - looks up a row of N_EMBED numbers for each for each token ID
        # think of it as a table: row 18 = the vector for 'T'
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)

        # positional embedding - looks up a row for each position
        # Tells the model WHERE in the sequence each token is
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        
        # Final layer norm
        self.norm = nn.LayerNorm(N_EMBED)

        # Output head - maps from N_EMBED numbers to vocab_size scores
        # One sscore per possible next token
        self.output_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding(x) # turns ids into vectors

        pos_emb =self.position_embedding(torch.arange(T, device=x.device)) # creates a vector for each postion

        x = tok_emb + pos_emb # x

        x = self.blocks(x)
        x = self.norm(x)
        out = self.output_head(x)
        return out
    
# Test

if __name__ == "__main__":

    print("Building Infinity-0 model...")
    model = InfinityTransformer(vocab_size=VOCAB_SIZE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Fake input - pretend we hae 2 sequences of 32 tokens each
    fake_input = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE)) # random integers between 0 and vocab size, shape (2, 32)
    print("Fake input shape:", fake_input.shape)

    # Forward pass
    output = model(fake_input)
    print(f"Output shape: {output.shape}")
    # should input 

    print()
    print("Output shape means:")
    print(f"  {output.shape[0]} sequences in the batch")
    print(f"  {output.shape[1]} positions per sequence")
    print(f"  {output.shape[2]} scores per position (one per vocab token)")
    print()
    print("Highest score at position 0 = model's guess for token after position 0")
    predicted_id = output[0, 0].argmax().item()
    print(f"Predicted token ID at position 0: {predicted_id}")
    print("(random weights so this guess is meaningless — training fixes that)")
