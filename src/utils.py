import torch
import numpy as np

def count_parameters(model):
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse_loss(x, x_hat):
    """Mean squared error loss."""
    total_prediction_loss = torch.sum((x - x_hat) ** 2) / x.size()[0]
    return total_prediction_loss


def relative_mse_loss(x, x_hat):
    """Relative mean squared error loss."""
    total_prediction_loss = torch.sum(((x - x_hat) ** 2) / (x ** 2)) / x.size()[0]
    return total_prediction_loss


def kl_div(mu, var, scale=1):
    """KL divergence with a standard normal distribution."""
    return torch.sum((var + mu ** 2) / (2 * scale)
                     + np.log(np.sqrt(scale))
                     - torch.log(torch.sqrt(var))
                     - 0.5)


def positive_identity(x):
    """Apply positive identity transformation."""
    y = x.clone()
    y[y > 0] = y[y > 0] + 1
    y[y < 0] = torch.exp(y[y < 0])
    return y
