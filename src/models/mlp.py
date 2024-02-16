import random

import torch
from torch import nn


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) neural network model.

    Args:
        in_features (int): The number of input features.
        num_classes (int): The number of output classes.
        num_hidden (int, optional): The number of hidden layers. Defaults to 3.
        hidden_size (int, optional): The number of neurons in each hidden layer. Defaults to 4.
    """

    def __init__(self, in_features: int, num_classes: int, num_hidden=3, hidden_size=4):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.in_features = in_features
        self.num_classes = num_classes
        layers = [
            nn.Linear(self.in_features, self.hidden_size),
            nn.ReLU(),
        ]

        # Add hidden layer
        for i in range(num_hidden):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_size, self.num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x.view(x.shape[0], -1)  # flatten along dim=1
        x = self.layers(x)
        return x


class GeneratorMLP(nn.Module):
    """
    A multi-layer perceptron (MLP) generator.

    Args:
        hidden_size (int): The number of neurons in the hidden layer.
        z_dim (int): The dimension of the noise vector.
        n_classes (int): The number of classes in the dataset.
        device (str): The device to run the model on.
    """


class GeneratorMLP(nn.Module):
    def __init__(self, hidden_size=10, z_dim=2, n_classes=2, device="cuda"):
        super(GeneratorMLP, self).__init__()
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.device = device
        self.layers = nn.Sequential(
            nn.Linear(z_dim + self.n_classes, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, n_classes),
        )

    def sample_z(self, batch_size=64):
        # z ~ N(0,I)
        # z = torch.randn(batch_size, self.z_dim).to(self.device)
        # z ~ U[0,1)
        z = torch.rand(batch_size, self.z_dim).to(self.device)
        return z

    def cond_pair_sample(self, batch_size=64):
        target = torch.randint(high=self.n_classes, size=(1, batch_size))
        # make one hot

        y = [sorted(list(range(self.n_classes)), key=lambda k: random.random()) for _ in range(batch_size)]
        y = torch.tensor(y)

        # we can turn this into a future loop for more than a pair of samples
        c1 = torch.zeros(batch_size, self.n_classes)
        c1[range(c1.shape[0]), y[:, 0].long()] = 1
        c1 = c1.to(self.device)

        c2 = torch.zeros(batch_size, self.n_classes)
        c2[range(c2.shape[0]), y[:, 1].long()] = 1
        c2 = c2.to(self.device)

        return c1, c2

    def decode(self, z, c):
        x = torch.cat((z, c), dim=-1)
        x = self.layers(x)
        return x

    def forward(self, c):
        # sample one common z
        z1 = self.sample_z()

        # sample some conditional labels
        c1, c2 = self.cond_pair_sample()

        # if self.cond_pair_sample gets turned into a loop, so will this
        # lazy
        # decode step 1
        x1_z1 = self.decode(z1, c1)

        # decode step 2 with same z but different c
        x2_z1 = self.decode(z1, c2)

        # sample a new z for the loss term that repels intra class examples

        # sample one common z
        z2 = self.sample_z()

        # if self.cond_pair_sample gets turned into a loop, so will this
        # lazy
        # decode step 1
        x1_z2 = self.decode(z2, c1)

        # decode step 2 with same z but different c
        x2_z2 = self.decode(z2, c2)

        # return the labels only, not the one hot vector as that's what most
        # pytorch environments expect
        return x1_z1, x2_z1, x1_z2, x2_z2, torch.argmax(c1, dim=1), torch.argmax(c2, dim=1)
