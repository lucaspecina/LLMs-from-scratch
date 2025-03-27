"""
Training and generation utilities for the GPT implementation
"""

from .trainer import train_model_simple, evaluate_model, calc_loss_batch, calc_loss_loader, plot_losses
from .generation import generate_text_simple, generate

__all__ = [
    'train_model_simple',
    'evaluate_model',
    'calc_loss_batch',
    'calc_loss_loader',
    'plot_losses',
    'generate_text_simple',
    'generate'
] 