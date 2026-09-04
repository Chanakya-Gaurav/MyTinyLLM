import torch
import argparse
import torch.nn as nn
from torch.nn import functional as F
import re
import time

##############################################################################
# tinyGPT.py
# A complete Transformer-based Language Model with:
# - Full Transformer Architecture (Attention + Feed-Forward blocks)
# - Multi-Head Self-Attention
# - Position Embeddings
# - Stacked Transformer Layers
# - Chat-like Interface for Natural Interaction
##############################################################################

##############################################################################
# --- 1. Hyperparameters ---
batch_size = 16   # Number of sequences processed in parallel (Default 16) 
block_size = 64   # Context window: how many characters to look back at (Default 128)<-Increase params and training time
n_embd = 128      # Embedding dimension: number of features per token (Default 128)
n_head = 4        # Number of attention heads (Default 4)
n_layer = 4       # Number of stacked transformer blocks (Default 4)
dropout = 0.1     # Dropout for regularization (Default = 0.1)

ffn_hidden = n_embd * 4  # Feed-forward hidden layer size (Default = 512)

learning_rate = 3e-4 #Default - 1e-3
CHECKPOINT_PATH = "tinyGPT_Checkpoint.pt"

# ------------ Training attributes
max_iters = 20000
eval_interval = 200
num_params = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

# Forece CPU for more precise embedding values (comment out if you want to use GPU for faster training)
device = "cpu"

##############################################################################
# --- 2. Data Preparation ---
# Using Shakespeare text as the training corpus
text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep...
""" * 100  # Repeat it to give the model more cycles to learn

text = """Once upon a time, in a small yard, there was a small 
daisy. The daisy had a name. Her name was Daisy. 
Daisy was very small, but she was also very happy.
One day, Daisy saw a dog. The dog was big and had a 
name too. His name was Max. Max liked to play in the yard. 
Daisy liked to watch Max play. Max and Daisy became friends.
Every day, Max would come to the yard to play. 
Daisy would watch and smile. They were very happy together. 
And even though Daisy was small, she knew that she had a 
big friend in Max.""" * 100  # Repeat it to give the model more cycles to learn

text = """The cat is in the kitchen. 
The cat is kicking the kitchen kettle. 
A friendly friend is friendly to a friend. 
The bright stars are brighter than the bright moon. 
This thin thing is thicker than this thin thing""" * 100
# Repeat it to give the model more "cycles" to learn

# Read a large dataset  Tiny Stories from Project Gutenberg (Hugging Face)
#with open('./corpus/TinyStories-valid.txt', 'r', encoding='utf-8') as f:
#    text_full = f.read()
#print("length of dataset in characters: ", len(text_full))

#text = text_full[:100000]  # Use only the first 100k characters for faster training

text = """hello hello hello hello world, 
this is a tiny tiny tiny tiny tiny llm
this is a tiny tiny tiny tiny tiny llm
actually a very tiny llm
this is only for education purposes.
hope hope this helps you understand how a transformer works.
"""
# Remove newlines and extra spaces to create a single line of text for training.
# Repeat 100 times to give the model more data to learn from, 
# which will help it learn the patterns better.
text = re.sub(r'\s*[\r\n]+\s*', ' ', text).strip() * 200

# tiny - 10 times, 
# this - 4 times, 

# hello - 8 times, 
# hope - 2 times,
# helps - 1 time, 
# how - 1 time, 

# Note that :
#   - 'i' follows 't' more frequently (5) than 'h' (3), so the model will learn to predict 'i' after 't' more often
#   - 'e' follows 'h' more frequently (5) than 'o' (3), so the model will learn to predict 'e' after 'h' more often

# For the tinyLLM.py try these combinations of inputs:
# 't' - expected output 'i' (5 times) and 
# 'h'  - expected output 'e' (5 times)

# For the tinyGPT.py try these combinations of inputs:
# 'hope t' - expected output 'h' (5 times)
# 'hope h' - expected output 'o' and 

# Character-level tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character to integer mapping and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # string -> list of ints
decode = lambda l: ''.join([itos[i] for i in l])  # list of ints -> string

# Convert entire text to tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Helper function to get random batches
# The model learns to predict the next character in the sequence, so it needs pairs of (current character, next character)
# This function will create a batch of: 
#   - Input  (x): a sequence of characters starting at a random index, and
#   - Target (y): the same sequence shifted by one character
#   - For example, if the input sequence is "hi there", the target will be "i there" (the next character after 'h' is 'i')
# The batch size determines how many of these sequences are processed in parallel during training, 
# and the block size determines how long each sequence is (how many characters the model looks back at).
# The batch_size will become the B (batch) dimension 
# and the block_size will become the T (time) dimension in the model,
# The C (channel) dimension will be the n_embd (embedding dimension) 

def get_batch(device=device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

##############################################################################
# --- 3. TRANSFORMER ARCHITECTURE COMPONENTS ---

# ============================================================================
# LAYER 1: Attention Head
# Purpose: Compute scaled dot-product attention for one "head"
# This allows the model to focus on different parts of the input
# ============================================================================
class Head(nn.Module):
    """Single head of self-attention mechanism"""
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        # Query: What am I looking for?
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # Key: What do I contain?
        self.key = nn.Linear(n_embd, head_size, bias=False)
        # Value: What information do I provide?
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Register buffer for causal mask (prevents looking at future tokens)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Query, Key, Value projections
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # This measures "affinities" between positions
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        
        # Apply causal mask: don't attend to future positions
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Convert scores to probabilities with softmax
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        
        # Apply attention weights to values
        out = wei @ v  # (B, T, head_size)
        return out


# ============================================================================
# LAYER 2: Multi-Head Attention
# Purpose: Run multiple attention heads in parallel and combine results
# This allows the model to attend to different types of patterns simultaneously
# ============================================================================
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    #(n_head, n_embd // n_head)
    #def __init__(self, num_heads, head_size):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        # Create multiple attention heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        # Project concatenated heads back to embedding dimension
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel and concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        # Project back to original dimension
        out = self.proj(out)
        out = self.dropout(out)
        return out


# ============================================================================
# LAYER 3: Feed-Forward Neural Network (MLPBlock)
# Purpose: Add non-linearity and expand the representation
# Structure: Linear -> ReLU -> Linear (with expansion in between)
# This helps the model learn complex patterns
# ============================================================================
class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, n_embd, ffn_hidden, dropout):
        super().__init__()
        # Expand to hidden dimension, then project back
        self.net = nn.Sequential(
            nn.Linear(n_embd, ffn_hidden),  # Expansion layer
            nn.ReLU(),                      # Activation Layer (Non-linearity)
            nn.Linear(ffn_hidden, n_embd),  # Contraction layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# LAYER 4: Transformer Block
# Purpose: Combine attention and feed-forward with residual connections
# This is the core building block that gets stacked multiple times
# Structure: [Attention + Residual + LayerNorm] -> [FFN + Residual + LayerNorm]
# ============================================================================
class TransformerBlock(nn.Module):
    """Single transformer block: Attention + Feed-Forward with residual connections"""
    def __init__(self, n_embd, n_head, block_size, dropout, ffn_hidden):
        super().__init__()
        # Multi-head attention component
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        # Feed-forward component
        self.ffn = FeedForward(n_embd, ffn_hidden, dropout)
        # Layer normalization (applied before attention and ffn for better stability)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Attention with residual connection and layer norm
        # Pre-normalization: normalize before applying the sublayer
        x = x + self.sa(self.ln1(x))
        
        # Feed-forward with residual connection and layer norm
        x = x + self.ffn(self.ln2(x))
        
        return x


# ============================================================================
# LAYER 5: Complete Transformer Language Model
# Purpose: Combines embeddings and stacked transformer blocks
# This is the full model that generates text
# ============================================================================
class TinyGPT(nn.Module):
    """Complete Transformer-based Language Model"""
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, ffn_hidden):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        self.ffn_hidden = ffn_hidden
        
        # ---- EMBEDDING LAYERS ----
        # Token Embedding: Convert token indices to dense vectors
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # Position Embedding: Give the model positional information
        # The model needs to know WHERE a token appears in the sequence
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # ---- TRANSFORMER BLOCKS ----
        # Stack multiple transformer blocks
        # Each block applies attention and feed-forward with residual connections
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size, dropout, ffn_hidden) for _ in range(n_layer)])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        
        # ---- OUTPUT LAYER ----
        # Linear projection from embedding space to vocabulary
        # This predicts the logits for the next token
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # ---- EMBEDDING PHASE ----
        # Convert token indices to embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        
        # Add positional information
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # Combine WHAT the token is with WHERE it is
        
        # ---- TRANSFORMER BLOCKS ----
        # Pass through all transformer blocks
        # Each block: [MultiHeadAttention + FeedForward + ResidualConnections]
        x = self.blocks(x)  # (B, T, n_embd)
        
        # Final layer normalization
        x = self.ln_f(x)
        
        # ---- PREDICTION PHASE ----
        # Project to vocabulary size to get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # ---- LOSS COMPUTATION (Training Only) ----
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # Flatten for cross-entropy loss
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate new tokens autoregressively"""
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Focus only on the last position
            
            # Apply temperature for controlling randomness
            logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_predict = torch.multinomial(probs, num_samples=1)
            
            # Append to context
            idx = torch.cat((idx, idx_predict), dim=1)
        
        return idx


##############################################################################
# --- 4. TRAINING FUNCTION ---
##############################################################################
def train_model(iters=max_iters, load_train=1, save_train=1, device="cpu"):
    """Train the TinyGPT model"""
    train_start_time = time.time()
    print("\n" + "="*70)
    print("INITIALIZING TINYGPT - COMPLETE TRANSFORMER LANGUAGE MODEL")
    print("="*70 + "\n")
    
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Unique Characters: {sorted(chars)}\n")
    
    # Initialize model
    model = TinyGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout, ffn_hidden).to(torch.device(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Track one query matrix (block 0, head 0) to observe weight updates over training.
    query_weight = model.blocks[0].sa.heads[0].query.weight
    query_weight_init = query_weight.detach().clone().to("cpu")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_params:,}\n")
    
    # Training loop
    print("="*70)
    print(f"TRAINING FOR {iters} ITERATIONS")
    print("="*70 + "\n")
    
    for iter in range(iters):
        xb, yb = get_batch(device)
        #Move the tensors to the same device as the model (GPU or CPU) for computation
        #xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % eval_interval == 0:
            q_now = query_weight.detach().to("cpu")
            # delta_norm = L2 (Euclidean length) distance between current and initial weights
            # how far the current query matrix moved from its initial values after training
            q_delta_norm = (q_now - query_weight_init).norm().item()
            q_sample = q_now[0, :5].tolist()
            q_sample = [round(v, 4) for v in q_sample]
            print(
                f"Iteration {iter:5d} | Loss: {loss.item():.4f} | "
                f"query.w[0,:5]: {q_sample} | delta_norm: {q_delta_norm:.6f}"
            )
    
    elapsed_time = time.time() - train_start_time
    print("\n" + "="*70)
    print("TRAINING COMPLETE - ElAPSED TIME: {:.2f} seconds".format(elapsed_time))
    print("="*70 + "\n")

    if save_train == 1:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "hyperparams": {
                "batch_size": batch_size,
                "block_size": block_size,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "dropout": dropout,
                "ffn_hidden": ffn_hidden,
                "learning_rate": learning_rate,
                "eval_interval": eval_interval,
                "max_iters": max_iters,
                "vocab_size": vocab_size,
                "num_params": num_params,
                "device_used": device.type,
                "training_time_seconds": elapsed_time,
            },
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"Checkpoint saved to: {CHECKPOINT_PATH}\n")
    else:
        print("Checkpoint not saved (save_train=0)\n")
    
    return model

##############################################################################
# --- 5. MAIN EXECUTION ---
##############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Train and run TinyLLM interactive generation.")
    parser.add_argument(
        "--iters",
        type=int,
        default=max_iters,
        help="Number of training iterations (integer >= 0).",
    )
    parser.add_argument(
        "--save_train",
        type=int,
        choices=[0, 1],
        default=0,
        help="Save checkpoint after training: 1=yes, 0=no.",
    )
    parser.add_argument(
        "--use_gpu",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use GPU for training: 1=yes, 0=no.",
    )

    args = parser.parse_args()
    if args.iters < 0:
        parser.error("--iters must be an integer >= 0")

    return args

if __name__ == "__main__":
    args = parse_args()

    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "TINYGPT - COMPLETE TRANSFORMER LANGUAGE MODEL".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)} for training.\n")

    # Train the model
    if args.iters < 200:
        eval_interval = max(1, args.iters // 10)

    model = train_model(iters=args.iters, save_train=args.save_train, device=device)
