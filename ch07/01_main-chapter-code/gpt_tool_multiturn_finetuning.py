# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# A file for supervised fine-tuning (SFT) a model for tool-using and multi-turn conversations

from functools import partial
from importlib.metadata import version
import json
import os
import re
import time
import urllib

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import from local files in this folder
from gpt_download import download_and_load_gpt2
from previous_chapters import (
    calc_loss_loader,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text
)


class ToolMultiTurnDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Markers for different parts of the conversation
        self.system_marker = "<|system|>"
        self.user_marker = "<|user|>"
        self.assistant_marker = "<|assistant|>"
        self.tool_call_marker = "<|tool_call|>"
        self.tool_result_marker = "<|tool_result|>"
        self.eot_marker = "<|endoftext|>"

        # Add these special tokens to the tokenizer
        # This approach works for tiktoken in learning mode
        # In production, you'd need to properly extend the tokenizer vocabulary
        self.special_tokens = [
            self.system_marker,
            self.user_marker,
            self.assistant_marker,
            self.tool_call_marker,
            self.tool_result_marker
        ]

        # Pre-tokenize and format conversations
        self.encoded_texts = []
        self.mask_indices = []  # Tracks where to apply masking (tool results, etc.)
        
        for conversation in data:
            formatted_conversation, mask_positions = self.format_conversation(conversation)
            encoded = tokenizer.encode(formatted_conversation)
            self.encoded_texts.append(encoded)
            
            # Convert mask positions from character indices to token indices
            # This is a simplified approach - in a real implementation you'd
            # need a more robust way to track token indices
            mask_token_indices = []
            for start, end in mask_positions:
                # This is approximate - in a real implementation you'd need to
                # align character and token indices precisely
                start_tokens = len(tokenizer.encode(formatted_conversation[:start]))
                end_tokens = len(tokenizer.encode(formatted_conversation[:end]))
                mask_token_indices.append((start_tokens, end_tokens))
            
            self.mask_indices.append(mask_token_indices)

    def format_conversation(self, conversation):
        """Format a conversation with system, user, assistant messages, and tools."""
        formatted_text = ""
        mask_positions = []  # (start, end) character positions to mask
        
        # Add system message if present
        if "system" in conversation and conversation["system"]:
            formatted_text += f"{self.system_marker}\n{conversation['system'].strip()}\n\n"
        
        # Add conversation turns
        for turn in conversation["conversation"]:
            if turn["role"] == "user":
                formatted_text += f"{self.user_marker}\n{turn['content'].strip()}\n\n"
            
            elif turn["role"] == "assistant":
                assistant_start = len(formatted_text)
                formatted_text += f"{self.assistant_marker}\n"
                
                # Process content which might include tool calls
                content = turn["content"]
                
                # If this assistant message has tool calls, format them specially
                if "tool_calls" in turn and turn["tool_calls"]:
                    # Split content to insert tool calls at the right positions
                    if isinstance(content, str):
                        formatted_text += f"{content.strip()}\n\n"
                    
                    # Add tool calls
                    for tool_call in turn["tool_calls"]:
                        tool_call_start = len(formatted_text)
                        formatted_text += f"{self.tool_call_marker}\n"
                        formatted_text += f"Tool: {tool_call['name']}\n"
                        formatted_text += f"Parameters: {json.dumps(tool_call['parameters'])}\n\n"
                        
                        # Add tool results - these should be masked during training
                        if "result" in tool_call:
                            result_start = len(formatted_text)
                            formatted_text += f"{self.tool_result_marker}\n"
                            formatted_text += f"{tool_call['result']}\n\n"
                            mask_positions.append((result_start, len(formatted_text)))
                else:
                    # Regular assistant message without tool calls
                    formatted_text += f"{content.strip()}\n\n"
        
        # End the conversation
        formatted_text += self.eot_marker
        return formatted_text, mask_positions

    def __getitem__(self, index):
        return self.encoded_texts[index], self.mask_indices[index]

    def __len__(self):
        return len(self.data)


def tool_multiturn_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Unpack batch - now contains encoded texts and mask positions
    items, mask_indices_list = zip(*batch)
    
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in items)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item, mask_indices in zip(items, mask_indices_list):
        new_item = item.copy()
        # Add an <|endoftext|> token if not already present
        if new_item[-1] != pad_token_id:
            new_item += [pad_token_id]
            
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        # Create inputs and targets
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        
        # Apply masking to targets where the model shouldn't predict
        # (tool results, system messages, etc.)
        for start_idx, end_idx in mask_indices:
            # Adjust indices to account for the +1 shift in targets
            # Ensure we're within bounds
            mask_start = max(0, min(start_idx, len(targets) - 1))
            mask_end = max(0, min(end_idx, len(targets) - 1))
            if mask_start < mask_end:
                targets[mask_start:mask_end] = ignore_index
        
        # Handle padding tokens in targets
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:  # If there are padding tokens
            if isinstance(indices, torch.Tensor) and indices.dim() > 0:
                targets[indices[1:]] = ignore_index  # Keep first pad token, mask others

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    """Download a file if it doesn't exist and load it."""
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """Plot training and validation losses."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plot_name = "tool-multiturn-loss-plot.pdf"
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)


def generate_tool_response(model, tokenizer, prompt, device, max_new_tokens=256, context_size=1024):
    """Generate a response from the model with potential tool calls."""
    # Encode the prompt
    input_ids = text_to_token_ids(prompt, tokenizer).to(device)
    
    # Generate tokens
    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=0.7,  # Using some temperature for creativity
            top_k=50,         # Using top-k sampling
            eos_id=tokenizer.encode("<|endoftext|>")[0] if "<|endoftext|>" in prompt else None
        )
    
    # Decode the output
    generated_text = token_ids_to_text(output_ids, tokenizer)
    
    # Extract just the model's response (everything after the prompt)
    response = generated_text[len(prompt):]
    
    return response


def main(test_mode=False):
    #######################################
    # Print package versions
    #######################################
    print()
    pkgs = [
        "matplotlib",  # Plotting library
        "tiktoken",    # Tokenizer
        "torch",       # Deep learning library
        "tqdm",        # Progress bar
        "tensorflow",  # For OpenAI's pretrained weights
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")

    #######################################
    # Download and prepare dataset
    #######################################
    file_path = "tool-multiturn-data.json"
    url = "URL_TO_YOUR_TOOL_MULTITURN_DATASET"  # Replace with actual URL
    
    # For testing purposes, create a sample dataset if file doesn't exist
    if not os.path.exists(file_path) or test_mode:
        # Create sample data with tool interactions
        sample_data = [
            {
                "system": "You are a helpful assistant with access to tools.",
                "conversation": [
                    {"role": "user", "content": "What's the weather in New York?"},
                    {
                        "role": "assistant", 
                        "content": "I'll check the weather for you.",
                        "tool_calls": [
                            {
                                "name": "get_weather",
                                "parameters": {"location": "New York"},
                                "result": "72°F, Partly Cloudy"
                            }
                        ]
                    },
                    {"role": "user", "content": "Thanks! And in San Francisco?"},
                    {
                        "role": "assistant",
                        "content": "Let me check San Francisco's weather.",
                        "tool_calls": [
                            {
                                "name": "get_weather",
                                "parameters": {"location": "San Francisco"},
                                "result": "65°F, Foggy"
                            }
                        ]
                    }
                ]
            },
            # Add a few more examples for testing
            {
                "system": "You are a helpful assistant with access to tools.",
                "conversation": [
                    {"role": "user", "content": "Can you calculate 235 × 489?"},
                    {
                        "role": "assistant",
                        "content": "I'll calculate that for you.",
                        "tool_calls": [
                            {
                                "name": "calculator",
                                "parameters": {"expression": "235 * 489"},
                                "result": "114915"
                            }
                        ]
                    }
                ]
            }
        ]
        
        # If testing, use this sample data
        if test_mode:
            data = sample_data
        else:
            # Otherwise, save it and try to download
            with open(file_path, "w") as f:
                json.dump(sample_data, f, indent=2)
            try:
                data = download_and_load_file(file_path, url)
            except:
                print("Could not download data. Using sample data instead.")
                data = sample_data
    else:
        # Load existing data
        data = download_and_load_file(file_path, url)

    # Split dataset into train, test, validation
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # Use very small subset for testing purposes
    if test_mode:
        train_data = train_data[:5]
        val_data = val_data[:2] if len(val_data) > 2 else val_data
        test_data = test_data[:2] if len(test_data) > 2 else test_data

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    # Create DataLoaders
    customized_collate_fn = partial(
        tool_multiturn_collate_fn, 
        device=device, 
        allowed_max_length=2048  # Increased from 1024 to handle longer conversations
    )

    num_workers = 0
    batch_size = 4  # Reduced from 8 due to potentially longer sequences

    torch.manual_seed(123)

    train_dataset = ToolMultiTurnDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = ToolMultiTurnDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # Load pretrained model
    #######################################

    # Small GPT model for testing purposes
    if test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 512,  # Increased from 120
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"
        CHOOSE_MODEL = "Small test model"

    # Code as it is used in the main chapter
    else:
        BASE_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 2048,  # Increased from 1024 for longer multi-turn conversations
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True         # Query-key-value bias
        }

        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        CHOOSE_MODEL = "gpt2-medium (355M)"

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        model.eval()
        model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    #######################################
    # Finetuning the model
    #######################################
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2

    torch.manual_seed(123)
    
    # For tool-based SFT, we need a sample conversation for evaluation during training
    sample_conversation = val_data[0]
    sample_input = train_dataset.format_conversation(sample_conversation)[0]
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=sample_input, tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    #######################################
    # Saving results
    #######################################
    print("Generating responses")
    for i, conversation in tqdm(enumerate(test_data), total=len(test_data)):
        # Get the last user message from the conversation
        last_user_turn = None
        for turn in reversed(conversation["conversation"]):
            if turn["role"] == "user":
                last_user_turn = turn
                break
        
        if last_user_turn:
            # Format the conversation up to the last user turn
            prompt_conversation = {
                "system": conversation.get("system", ""),
                "conversation": []
            }
            
            # Include all turns up to the last user turn
            for turn in conversation["conversation"]:
                prompt_conversation["conversation"].append(turn)
                if turn == last_user_turn:
                    break
            
            # Format and generate response
            prompt, _ = train_dataset.format_conversation(prompt_conversation)
            
            response = generate_tool_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=256,
                context_size=BASE_CONFIG["context_length"]
            )
            
            # Store the model's response
            test_data[i]["model_response"] = response

    # Save responses
    test_data_path = "tool-multiturn-responses.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)
    print(f"Responses saved as {test_data_path}")

    # Save model
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-tool-multiturn-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune a GPT model for tool-using and multi-turn conversations"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode with a small dataset. "
              "Otherwise, it runs the model with the full dataset.")
    )
    args = parser.parse_args()

    main(args.test_mode) 