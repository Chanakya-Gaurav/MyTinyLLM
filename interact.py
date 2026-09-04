import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from tinyGPT import TinyGPT

CHECKPOINT = "tinyGPT_Checkpoint.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint '{path}' not found. Run tinyGPT.py training first to create it."
        )

    checkpoint = torch.load(path, map_location=device)

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    hp = checkpoint["hyperparams"]

    model = TinyGPT(
        vocab_size=hp["vocab_size"],
        n_embd=hp["n_embd"],
        n_head=hp["n_head"],
        n_layer=hp["n_layer"],
        block_size=hp["block_size"],
        dropout=hp["dropout"],
        ffn_hidden=hp["ffn_hidden"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, stoi, itos, checkpoint["hyperparams"]


def chat_interface(model, stoi, itos):
    print("\nInstructions:")
    print("  - Enter a prompt to generate text (or press Enter for random start)")
    print("  - Type 'quit' to exit\n")
    print("  - Try prompts like 'h', 't', 'hope h', and 'hope t' to compare with TinyGPT")
    print("  -   Expected output for 't': ['i', 'h', 'r']")
    print("  -   Expected output for 'h': ['e', 'o', 'i']")
    print("  -   Expected output for 'hope t': 'h'")
    print("  -   Expected output for 'hope h': 'o'")


    temperature = 1.0
    max_tokens = 50

    sorted_chars = sorted(stoi.keys())
    fallback_char = " " if " " in stoi else sorted_chars[0]
    fallback_idx = stoi[fallback_char]

    def encode_prompt_safe(prompt):
        return [stoi[c] if c in stoi else fallback_idx for c in prompt]

    def decode(ids):
        return "".join([itos[i] for i in ids])

    while True:
        try:
            print("-" * 70)
            user_input = input("You: ").strip()

            if user_input.lower() == "quit":
                print("\nThank you for chatting with TinyGPT. Goodbye!\n")
                break

            if user_input.lower() == "settings":
                print("\n--- Generation Settings ---")
                print(f"Current Temperature: {temperature}")
                print(f"Current Max Tokens: {max_tokens}")
                try:
                    temp_input = input("New temperature (or press Enter to skip): ").strip()
                    if temp_input:
                        temperature = float(temp_input)
                    tokens_input = input("New max tokens (or press Enter to skip): ").strip()
                    if tokens_input:
                        max_tokens = int(tokens_input)
                    print()
                except ValueError:
                    print("Invalid input, keeping current settings.\n")
                continue

            if user_input:
                context_ids = encode_prompt_safe(user_input)
                context = torch.tensor([context_ids], dtype=torch.long, device=device)
            else:
                print("(Empty input, generating from random start...)")
                context = torch.tensor([[fallback_idx]], dtype=torch.long, device=device)

            generated_ids = model.generate(
                context,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )[0].tolist()

            context_len = context.shape[1]
            new_generated_ids = generated_ids[context_len:]
            print(f">> TinyGPT (predicted next character): {decode(new_generated_ids[0:1])}")
            print(f">> TinyGPT (complete generation): {decode(generated_ids)}\n")

        except KeyboardInterrupt:
            print("\n\n(Chat interrupted by user.)\n")
            break


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*70 + "╗")
    print("║" + " "*70 + "║")
    print("║" + "WELCOME TO TINYGPT - INTERACTIVE TEXT GENERATION".center(70) + "║")
    print("║" + " "*70 + "║")
    print("╚" + "="*70 + "╝")

    #print("\n" + "=" * 70)
    #print("WELCOME TO TINYGPT - INTERACTIVE TEXT GENERATION")
    #print("=" * 70)

    print("\n" + "-" * 70 )
    print(f"Loading checkpoint from {CHECKPOINT} on {device}...")
    model, stoi, itos, hp = load_checkpoint(CHECKPOINT)
    print("Model loaded... ")
    print(f"  - Vocab size         : {hp['vocab_size']}")
    print(f"  - No. of Parameters  : {hp['num_params']}")
    print(f"  - Device used        : {hp['device_used']}")
    print(f"  - Training iterations: {hp['max_iters']}")
    print(f"  - Training time (s)  : {hp['training_time_seconds']:.2f}\n")
    print("-" * 70 + "\n")
    chat_interface(model, stoi, itos)
