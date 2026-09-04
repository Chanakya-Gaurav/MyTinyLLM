import os
import re
import time
import argparse
import torch

##############################################################################
# Credits: This code is inspired by Andrej Karpathy's "nanoGPT" project
# The original code can be found at: https://github.com/karpathy/nanoGPT

# pyTorch is a popular deep learning library that provides tools for 
# building and training neural networks.
# Documentation: https://docs.pytorch.org/docs/main/

# We are using the torch.nn module to define our model architecture 
# and the torch.nn.functional module for loss calculation and activation functions.
# Documentation: https://docs.pytorch.org/docs/2.12/nn.html

# Some aspects of the Transformer are implemented using PyTorch's built-in functionalities
# Documentation: https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
##############################################################################

##############################################################################
# --- 1. Get the Corpus ---
##############################################################################

## We need to convert text into numbers. In this tiny model, we treat each character as a "token."
# Our "Small Corpus"
text = """hello hello hello hello world, 
this is a tiny tiny tiny tiny tiny llm
this is a tiny tiny tiny tiny tiny llm
actually a very tiny llm
this is only for education purposes.
hope hope this helps you understand how a transformer works.
"""
# Remove newlines and extra spaces to create a single line of text for training.
# Repeat 200 times to give the model more data to learn from, 
# which will help it learn the patterns better.
text = re.sub(r'\s*[\r\n]+\s*', ' ', text).strip() * 200

# tiny - 6 times, 
# this - 4 times, 

# hello - 4 times, 
# helps - 1 time, 
# hope - 2 times,
# how - 1 time, 

# Note that :
#   - 'i' follows 't' more frequently (6) than 'h' (3), so the model will learn to predict 'i' after 't' more often
#   - 'e' follows 'h' more frequently (5) than 'o' (3), so the model will learn to predict 'e' after 'h' more often

# For the tinyLLM.py try these combinations of inputs:
# 't' - expected output 'i' (6 times) or 'h' (3 times) or 'r'
# 'h'  - expected output 'e' (4 times) or 'o' (3 times) or 'i' (3 times) or 'a' (2 times) or 'l' (2 times) or 's' (1 time) or 'w' (1 time) or 'p' (1 time)

# For the tinyGPT.py try these combinations of inputs:
# 'hope h' - expected output 'o' and 
# 'hope t' - expected output 'h' (5 times)

##############################################################################
# --- 2. Set the Hyperparameters ---
##############################################################################
block_size = 20
num_params = 0
learning_rate = 1e-3 #Default - 3e-4

# ----- Training Parameters -----
device = "cpu" # Change to "cuda" if you have a compatible GPU and want to train faster
# Note: Training on a GPU can significantly speed up the process, especially for larger models 
# but at the cost of precision in the embedding values. 
# If you want to see more precise embedding values, use "cpu" for training.
# CPU uses FP64 precision which gives more precise embedding values, 
# while GPU uses FP16 precision which is faster but less precise.
eval_interval = 200
max_iters = 3000

# Introduce some reproducibility by setting a random seed for PyTorch.
# This will ensure that any random operations performed by PyTorch after this line will 
# produce the exact same results every time we run this code
torch.manual_seed(1337)

CHECKPOINT_PATH = "tinyLLM_Checkpoint.pt"

##############################################################################
# --- 3. Get the Tokens ---
##############################################################################
chars = sorted(list(set(text))) 
# Unique characters in the corpus, sorted alphabetically. This forms our vocabulary of tokens.
# vocab_size is the number of unique characters in the text, which determines the size of our model's "brain" (embedding layer).
vocab_size = len(chars)
if block_size > vocab_size:
    block_size = vocab_size - 1

##############################################################################
# --- 4. Generate token IDs ---
##############################################################################
# # 2. Mapping characters to integers as their token index.
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] 
# encode: will associate a index to each token (character) in the string, 
# converting it to a python list of integers
decode = lambda l: ''.join([itos[i] for i in l]) 
# decode: will convert a list of integers back to a string

# Convert entire text to a tensor
# A numeric array of token IDs of all the tokens in the input text that 
# the model can understand and learn from.
data = torch.tensor(encode(text), dtype=torch.long)

##############################################################################
## The "Brain" (BiGram Model)
##############################################################################
# ## A bigram model works on a very simple rule: The probability of the next character 
## depends only on the current character
## This model is basically a lookup table where:
#### Rows represent the character you are looking at right now
#### Columns represent the the score for possible next characters   
#### It looks at one character and tries to guess the most likely 
#### next one using an Embedding Layer.

## F.cross_entropy: A teacher grading a spelling test. 
#### It compares the model's guess (logits) to the correct answer (targets) 
#### and gives a score (loss) that tells the model how well it did. 
#### The model then uses this score to adjust its "brain" (parameters) to do better next time.

#### The forward pass: The model's "thinking or learning process." 
#### It takes in a sequence of characters (idx) and optionally the correct next characters (targets for training).
#### If targets are provided, it calculates the loss; otherwise, it just returns the logits for the next character predictions.
#### For every single position in the B-batch and T-time, there are C=vocab_size scores predicting what comes next
#### This dimension C holds the logits (raw prediction scores) for every possible next character in the vocabulary

#### The generate method: The model's "imagination or inference engine."
#### It takes a starting point (context) and generates new characters based on what it has learned.

import torch.nn as nn
from torch.nn import functional as F

class BiGramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Create a lookup table (embedding layer) that maps each token index to a vector of size vocab_size.
        # Initially this embedding is random, but during training, it will learn to represent the relationships between characters.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # This is the forward-pass Computation, which is the model's core prediction step
    # Training: It takes in a sequence of token indices (idx) 
    # and the corresponding target indices (targets) and compute cross-entropy loss
    # Inference: Returns the logits for the next character predictions.
    # This function is implicitly called when the model instance is called with input data.
    # During training with the targets provided model(xb, yb), it computes the loss. 
    # During inference without targets self(idx) called from generate, it just returns the logits.
    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) 
        ## The shape of idx is (B, T) where B is the batch size and T is the sequence length.
        ## The token_embedding_table is a lookup table that maps each token index to a vector of size vocab_size.
        ## Therefore, for each (B, T) it will return a vector of size (C), of size vocab_size.
        ## This is the prediction step where the model looks at the 
        ## current token and produces a score for each possible next token in the vocabulary.
        ## resulting in a tensor of shape (B, T, C)
        #### Note: In the initial run with a random state, these logits are just random scores, 
        ####       but as the model trains, it will learn to produce more meaningful scores that reflect
        ####       the likelihood of each next character based on the training data. 

        if targets is None:
            # This is the inference mode, where we just want to get the logits for the next character predictions.
            loss = None
        else:
            #This is the training mode, where we compute the loss by comparing the predicted logits to the target indices.
            B, T, C = logits.shape
            ## cross_entropy expects a 2D input
            ## where each row is an example and each column is a score for a category
            ## logits = logits.view(B*T, C) - flattens 3D tensor to 2D by combining batch and time dimensions
            ## targets = targets.view(B*T) - flattens targets to a 1D tensor of the same length as the number of examples in logits
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            ## Note: Since the target the occurance of a character following another character 
            ## is not deterministic, the model will learn to assign probabilities to the next characters 
            ## based on their frequency in the training data.
            ## This is where the learning and adjusting the parameter weights is happening based on the loss calculated from the model's predictions and the actual targets.
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # Use the forward method in the inference mode to get the logits for the next character predictions,
        # The logits returned contains the scores for all possible next characters in the vocabulary 
        # for each position in the input sequence.
        # The softmax function is applied to these logits to convert them into probabilities, 
        # which represent the likelihood of each possible next character.
        for _ in range(max_new_tokens):
            # Get the logits for the next character predictions based on the current context (idx).
            logits, loss = self(idx)

            #take only the predictions corresponding to the last token in the current context,
            logits = logits[:, -1, :] # focus only on the last time step

            # Converts raw scores into probabilities for the next character prediction.
            probs = F.softmax(logits, dim=-1) # get probabilities
            
            # Pick the next token index using the model’s probability distribution
            idx_predict = torch.multinomial(probs, num_samples=1)

            # Append the sampled index to the current context (idx) to form the new context for the next prediction. 
            idx = torch.cat((idx, idx_predict), dim=1) # append sampled index
        return idx

##############################################################################
# --- 5. TRAINING FUNCTION ---
##############################################################################
def get_chunks(data):
    # Generate random starting indices for the batch
    ix = torch.randint(0, len(data) - block_size - 1, (1,))

    # Create input (x) and target (y) batches by slicing the data at the generated indices
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return xb, yb

def train_model(iters=max_iters, load_train=0, save_train=1):
    ## Training the Model
    ## We show the model the text over and over again, 
    ## and it "learns" which letters usually follow each other 
    ## (like 'h' followed by 'e').
    ## AdamW: A smart way to adjust the model's "brain" based on the loss.

    if load_train and os.path.exists(CHECKPOINT_PATH) and (checkpoint := torch.load(CHECKPOINT_PATH, map_location='cpu')) is not None:
        print("\n" + "=" * 70)
        print("LOADING CHECKPOINT ...")
        print("-" * 70 + "\n")

        hp = checkpoint["hyperparams"]
        model = BiGramLanguageModel(vocab_size=hp["vocab_size"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval() # Set the model to evaluation mode after loading the checkpoint

        print(f"Checkpoint loaded from: {CHECKPOINT_PATH}")
        print(f"  - Vocab size         : {hp['vocab_size']}")
        print(f"  - No. of Parameters  : {hp['num_params']}")
        print(f"  - Training iterations: {hp['max_iters']} \n")
        print(f"  - Device used        : {hp.get('device_used', 'unknown (legacy checkpoint)')}")
        print("-" * 70 + "\n")

    else:
        if device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)} for training.\n")
        else:
            print("Using CPU for training.\n")

        print(f"Unique characters: {chars}\n")
        print(f"Vocabulary Size: {vocab_size}")

        # Initialize the BiGram Language Model and count the number of parameters
        model = BiGramLanguageModel(vocab_size).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"No. of Parameters: {num_params}")
        print("*Note: In this model the number of parameters is equal to vocab_size^2 because each token has a vector of size vocab_size, and there are vocab_size tokens.")
        print("-" * 70 + "\n")

        # This is the trainer who will adjust the model's parameters based 
        # on the loss calculated from the model's predictions and the actual targets.
        # AdamW -> Adaptive Moment Estimation with Weight Decay,
        # which is a popular optimization algorithm used by GPT, BERT, etc.
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Inspect the initial embedding for 'h' before training
        h_id = stoi['h'] if 'h' in stoi else None
        h_emb_before_rounded = None
        if h_id is not None:
            embedding_device = model.token_embedding_table.weight.device
            h_emb_before = model.token_embedding_table(
                torch.tensor([h_id], dtype=torch.long, device=embedding_device)
            ).squeeze(0)
            h_emb_before_rounded = [round(v, 3) for v in h_emb_before.tolist()]

        # Simple training loop
        print("=" * 70 + "\n")
        print(f"Training the BiGram Language Model for {iters} Iterations\n")
        print("=" * 70 + "\n")

        train_start_time = time.time()
        for steps in range(iters): 
            # Sample a batch of data 
            #--------------------- highly simplified ------------------
            # In this case the xb and yb will be exactly the same in every training step
            xb, yb = data[:-1].unsqueeze(0), data[1:].unsqueeze(0)

            #Move the tensors to the same device as the model (GPU or CPU) for computation
            xb, yb = xb.to(device), yb.to(device)

            #--------------------- random chunks ------------------
            # In this case the xb and yb will be different slice of the data in every training step, 
            # which will help the model learn better by seeing different parts of the text.
            #### For using the simple training data comment this
            #xb, yb = get_chunks(data)
            #xb, yb = xb.to(device), yb.to(device)

            ## xb is all elements of data except the last one, 
            ## yb is all elements of data except the first one.
            ## for a sequence [1, 2, 3, 4], 
            ## xb becomes [1, 2, 3] 
            ## and yb becomes [2, 3, 4]
            ## At position 0, the input is 1 and the target is 2
            ## .unsqueeze(0) inserts a new dimension at index 0
            ## The final shape of xb and yb becomes (1, T), where B=1 and T is the length of the sequence minus one.

            # Evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            # Backpropagation: The model's "learning process" where it adjusts its parameters based on the loss.
            loss.backward()
            optimizer.step()

            if steps % eval_interval == 0:
                elapsed_time = time.time() - train_start_time
                print(f"Iteration {steps:5d} | Loss: {loss.item():.4f} | Elapsed time: {elapsed_time:.2f}s")

        # Inspect learned embedding matrix dimensions and a few character embeddings.
        print("-" * 70 + "\n")
        print("\nLearned Embedding Inspection\n")
        
        embedding_weight = model.token_embedding_table.weight
        print(f"Embedding matrix shape (vocab_size x embedding_dim): {tuple(embedding_weight.shape)}\n")
        print("-" * 70 + "\n")

        # Inspect the learned embedding for 'h' after training 
        if h_id is not None:
            h_emb = model.token_embedding_table(
                torch.tensor([h_id], dtype=torch.long, device=embedding_device)
            ).squeeze(0)
            h_emb_rounded = [round(v, 3) for v in h_emb.tolist()]
            print(f"Initial embedding for 'h' (id={h_id}): {h_emb_before_rounded}\n")
            print(f"Learned embedding for 'h' (id={h_id}): {h_emb_rounded}")
    
        elapsed_time = time.time() - train_start_time
        print("\n" + "="*70)
        print("TRAINING COMPLETE - ElAPSED TIME: {:.2f} seconds".format(elapsed_time))
        print("="*70 + "\n")

    if save_train == 1:
        print("\n" + "=" * 70)
        print("Training Complete. Saving Checkpoint ...")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "hyperparams": {
                "vocab_size": vocab_size,
                "num_params": num_params,
                "max_iters": iters,
                "device_used": device.type,
            },
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"Checkpoint saved to: {CHECKPOINT_PATH}\n")
        print("=" * 70 + "\n")

    return model

##############################################################################
# --- 6. TEXT GENERATION FUNCTION ---
##############################################################################
## After training, we can ask the model to generate new text by  
## giving it a starting character
## model.generate: Predicting the next word in a text message.
#### The model's "imagination" engine. 
#### It takes a starting point (context) and generates new characters 
#### based on what it has learned.


# Starting with 'h'
#context = torch.tensor([[stoi['h']]], dtype=torch.long) 
#generated_idx = model.generate(context, max_new_tokens=100)

# start with the first character index
#context = torch.zeros((1, 1), dtype=torch.long) 
#print(decode(model.generate(context, max_new_tokens=50)[0].tolist()))

# Interactive generation loop
fallback_char = ' ' if ' ' in stoi else chars[0]
fallback_idx = stoi[fallback_char]

def encode_prompt_safe(prompt):
    # Replace unknown characters with a known fallback to avoid KeyError.
    return [stoi[c] if c in stoi else fallback_idx for c in prompt]

def interactive_generation(model):
#    print("\n============= Interactive Text Generation =============\n")
    print("\n" + "=" * 70)
    print("WELCOME TO TINLLM - INTERACTIVE TEXT GENERATION")
    print("=" * 70)
    print("\nInstructions:")
    print("  - Enter a prompt to generate text (or press Enter for random start)")
    print("  - Type 'quit' to exit\n")
    print("  - Try prompts like 'h', 't', 'hope h', and 'hope t' to compare with TinyGPT")
    print("  -   Expected output for 't': ['i', 'h', 'r']")
    print("  -   Expected output for 'h': ['e', 'o', 'i']")
    print("  -   Expected output for 'hope t': 'h'")
    print("  -   Expected output for 'hope h': 'o'")
    print("=" * 70 + "\n")

    embedding_device = model.token_embedding_table.weight.device
    while True:
        try:
            print("-" * 70)
            user_input = input("You: ").strip()

            if user_input.strip().lower() == "quit":
                print("Exiting.")
                break

            if user_input:
                context_ids = encode_prompt_safe(user_input)
                context = torch.tensor([context_ids], dtype=torch.long, device=embedding_device)
            else:
                context = torch.tensor([[fallback_idx]], dtype=torch.long, device=embedding_device)

            context_len = context.shape[1]
            generated_ids = model.generate(context, max_new_tokens=50)[0].tolist()
            new_generated_ids = generated_ids[context_len:]

            print(f">> TinyLLM (predicted next character): {decode(new_generated_ids[0:1])}")
            print(f">> TinyLLM (complete generation): {decode(generated_ids)}\n")

        except KeyboardInterrupt:
            print("\n\n(Chat interrupted by user.)\n")
            break

##############################################################################
# --- 7. MAIN EXECUTION ---
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
        "--load_train",
        type=int,
        choices=[0, 1],
        default=0,
        help="Load checkpoint before training: 1=yes, 0=no.",
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
        default=1,
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
    print("║" + "TINY LARGE LANGUAGE MODEL".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # Train the model
    if args.iters < 200:
        eval_interval = max(1, args.iters // 10)

    model = train_model(iters=args.iters, load_train=args.load_train, save_train=args.save_train)

    # Start interactive generation
    interactive_generation(model)