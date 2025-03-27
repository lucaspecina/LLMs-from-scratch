"""
LLMs from Scratch - A modular implementation of a GPT model
"""

from .models.transformer import GPTModel, TransformerBlock, LayerNorm, GELU, FeedForward
from .models.attention import MultiHeadAttention
from .data.dataset import GPTDatasetV1, create_dataloader_v1
from .training.trainer import train_model_simple, evaluate_model, calc_loss_batch, calc_loss_loader, plot_losses
from .training.generation import generate_text_simple, generate
from .utils.tokenizer import text_to_token_ids, token_ids_to_text
from .utils.model_loader import load_weights_into_gpt

__all__ = [
    'GPTModel',
    'TransformerBlock',
    'LayerNorm',
    'GELU',
    'FeedForward',
    'MultiHeadAttention',
    'GPTDatasetV1',
    'create_dataloader_v1',
    'train_model_simple',
    'evaluate_model',
    'calc_loss_batch',
    'calc_loss_loader',
    'plot_losses',
    'generate_text_simple',
    'generate',
    'text_to_token_ids',
    'token_ids_to_text',
    'load_weights_into_gpt'
] 