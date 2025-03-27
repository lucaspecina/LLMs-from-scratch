import torch
import tiktoken
from .models import GPTModel
from .data import create_dataloader_v1
from .training import train_model_simple, generate
from .utils import text_to_token_ids, token_ids_to_text

def main():
    # Model configuration
    model_config = {
        "vocab_size": 50257,  # GPT-2 vocabulary size
        "emb_dim": 768,       # Embedding dimension
        "context_length": 128, # Maximum context length
        "n_heads": 12,        # Number of attention heads
        "n_layers": 6,        # Number of transformer layers
        "drop_rate": 0.1,     # Dropout rate
        "qkv_bias": False     # Whether to use bias in QKV projections
    }

    # Training configuration
    train_config = {
        "batch_size": 4,
        "max_length": 64,
        "stride": 32,
        "num_epochs": 5,
        "eval_freq": 100,
        "eval_iter": 5,
        "learning_rate": 0.0004,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Sample text for training (replace with your own text)
    train_text = """
    This is a sample text for training the GPT model.
    You should replace this with your own training data.
    The model will learn to predict the next token in the sequence.
    """

    # Create data loaders
    train_loader = create_dataloader_v1(
        train_text,
        batch_size=train_config["batch_size"],
        max_length=train_config["max_length"],
        stride=train_config["stride"]
    )

    # Initialize model
    model = GPTModel(model_config).to(train_config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"])

    # Training loop
    print("Starting training...")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # No validation data in this example
        optimizer=optimizer,
        device=train_config["device"],
        num_epochs=train_config["num_epochs"],
        eval_freq=train_config["eval_freq"],
        eval_iter=train_config["eval_iter"],
        start_context="Once upon a time",
        tokenizer=tokenizer
    )

    # Generate some sample text
    print("\nGenerating sample text...")
    context = "The story begins"
    encoded = text_to_token_ids(context, tokenizer).to(train_config["device"])
    
    with torch.no_grad():
        generated_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=model_config["context_length"],
            temperature=0.7,
            top_k=40
        )
    
    generated_text = token_ids_to_text(generated_ids, tokenizer)
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main() 