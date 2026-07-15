import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            # forward pass layer by layer and record each layer stats
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean_val = round(x.mean().item(), 4)
                    std_val = round(x.std().item(), 4)
                    dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    stats.append({'mean': mean_val, 'std': std_val, 'dead_fraction': dead_frac})

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad() 
        output = model(x) # forward pass 
        loss = nn.MSELoss()(output, y) # compare output of forward pass and output tensors 
        loss.backward()
        stats = []
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                mean_val = round(grad.mean().item(), 4)
                std_val = round(grad.std().item(), 4)
                norm_val = round(torch.norm(grad).item(), 4)
                stats.append({'mean': mean_val, 'std': std_val, 'norm': norm_val})
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for s in activation_stats:
            if s['dead_fraction'] > 0.5:
                return 'dead_neurons'

        for g in gradient_stats:
            if g['norm'] > 1e3:
                return 'exploding_gradients'

        if gradient_stats[-1]['norm'] < 1e-5 :
            return 'vanishing_gradients'

        for g in activation_stats:
            if g['std'] < 0.1:
                return 'vanishing_gradients'
            if g['std'] > 10.0:
                return 'exploding_gradients'

        return 'healthy'

