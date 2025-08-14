import torch
import torch.nn as nn
import biotrainer.utilities as utils

from .biotrainer_model import BiotrainerModel


# Convolutional neural network (two convolutional layers)
class CNN(BiotrainerModel):
    def __init__(
            self, 
            n_classes: int, 
            n_features: int,
            **kwargs,
    ):
        super(CNN, self).__init__()
        
        model_params = kwargs["model_params"]
        dropout_rate = model_params["dropout_rate"]
        kernel_sizes = model_params["kernel_sizes"]
        padding = model_params["padding"]
        hidden_dims = model_params["hidden_dims"]
        last_layer_FNN = model_params["last_layer_FNN"]

        dims = [n_features] + hidden_dims + [n_classes]

        layers = []
        for i in range(len(dims)):
            if i == len(dims) - 2:  # Last layer
                if last_layer_FNN:
                    layers.append(nn.Linear(dims[i], dims[i + 1]))
                else:
                    layers.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=kernel_sizes[i], padding=padding[i]))
                break
            else:
                layers.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=kernel_sizes[i], padding=padding[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

        self.net = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (9 for conservation)
        """

        # Calculate mask
        mask = (x.sum(dim=-1) != utils.SEQUENCE_PAD_VALUE).unsqueeze(1).unsqueeze(3)  # Shape: (B, 1, L, 1)

        x = x.permute(0, 2, 1).unsqueeze(3)  # Shape: (B, F, L, 1)

        x = self.net(x)
        x = x * mask  # Apply mask

        # Remove the last dimension and permute back
        x = x.squeeze(3).permute(0, 1, 2)  # Shape: (B, L, N)

        return x
