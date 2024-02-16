#!/usr/bin/env python3


from torch import nn


class LogisticRegression(nn.Module):
    """Logistic regression model

    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
    """

    def __init__(self, input_dim, output_dim):
        """Initialize the model."""
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass."""
        x = x.view(x.shape[0], self.input_dim)
        y = self.linear(x)
        return self.softmax(y)
