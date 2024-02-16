#!/usr/bin/env python3

"""This file implements a binary kernel SVM classifier."""

from sklearn import datasets
import torch
import numpy as np
from torch import nn
import math


import torch
import torch.optim as optim
import torch.nn as nn
import abc

from torchmetrics import HingeLoss


class AbstractPrimalSVM(nn.Module, metaclass=abc.ABCMeta):
    """Abstract class for a primal SVM classifier."""

    def __init__(self, num_features: int, C: float, num_classes=2):
        super(AbstractPrimalSVM, self).__init__()
        self.linear = nn.Linear(num_features, num_classes, bias=True)
        self.num_classes = num_classes
        self.C = C

        self.hinge_loss = HingeLoss("multiclass", num_classes=num_classes)

    def forward(self, X):
        X = X.view(X.shape[0], -1)
        X_transformed = self.feature_transform(X)
        scores = self.linear(X_transformed).squeeze()
        return scores

    def feature_transform(self, X):
        return X

    def loss(self, scores, y):
        if y.dim() > 1:
            y = y.argmax(dim=1)
        hinge_loss = self.hinge_loss(scores, y)
        return hinge_loss + self.C * self.linear.weight.norm(2) / y.shape[0]

    def predict(self, X):
        return torch.softmax(self(X), dim=1).argmax(dim=1)


class PrimalLinearSVM(AbstractPrimalSVM):
    """Linear SVM classifier."""

    def __init__(self, num_features: int, C=1.0, num_classes=2):
        super(PrimalLinearSVM, self).__init__(num_features, C=C, num_classes=num_classes)


class PrimalPolySVM(AbstractPrimalSVM):
    """Polynomial SVM classifier.""" ""

    def __init__(self, num_features: int, num_classes: int, degree=3, C=1.0):
        super(PrimalPolySVM, self).__init__(num_features * degree, num_classes=num_classes, C=C)
        self.degree = degree
        self.C = C

    def feature_transform(self, X):
        return self.polynomial_features(X)

    def polynomial_features(self, X):
        return torch.cat([X**i for i in range(1, self.degree + 1)], 1)


class PrimalRBFSVM(AbstractPrimalSVM):
    """RBF SVM classifier."""

    def __init__(self, num_features: int, num_classes: int, num_fourier_features: int, gamma=1.0, C=1.0):
        super(PrimalRBFSVM, self).__init__(num_fourier_features, num_classes=num_classes, C=C)
        self.gamma = gamma
        self.num_fourier_features = num_fourier_features

        # Randomly sample weights and biases for the Fourier features
        self._omega = torch.randn(num_features, num_fourier_features) * math.sqrt(2 * gamma)
        self._bias = torch.rand(num_fourier_features) * 2 * math.pi
        self.register_buffer("omega", self._omega)
        self.register_buffer("bias", self._bias)

    def random_fourier_features(self, X):
        """Compute the random Fourier features for the RBF kernel."""
        # Generate the random Fourier features
        projection = X @ self.omega + self.bias
        return np.sqrt(2.0 / self.num_fourier_features) * torch.cos(projection)

    def feature_transform(self, X):
        return self.random_fourier_features(X)


if __name__ == "__main__":
    # Example usage:
    import matplotlib.pyplot as plt
    import numpy as np

    import numpy as np
    import matplotlib.pyplot as plt

    def plot_decision_boundary(model, X, y):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = torch.argmax(model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])), dim=1)
        Z = Z.reshape(xx.shape).detach().numpy()

        cb = plt.contourf(xx, yy, Z, alpha=0.8, cmap="PRGn", levels=21)
        plt.colorbar(cb)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", linewidth=1)

        weights = model.linear.weight.detach().numpy()
        biases = model.linear.bias.detach().numpy()

        x_vals = np.linspace(x_min, x_max, 400)

        for i in range(weights.shape[0]):
            w = weights[i]
            b = biases[i]
            y_vals = -(w[0] / w[1]) * x_vals - b / w[1]
            plt.plot(x_vals, y_vals, "--", label=f"Hyperplane {i}")

        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.show()

    # Assume model, X_data, and y_data are defined and the model is trained
    # plot_decision_boundary(model, X_data, y_data)

    # def plot_decision_boundary(model, X, y):
    #     h = 0.02
    #     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    #     Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))[:, 0]
    #     Z = Z.reshape(xx.shape).detach().numpy()

    #     cb = plt.contourf(xx, yy, Z, alpha=0.8, cmap="PRGn", levels=21)
    #     plt.colorbar(cb)
    #     plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", linewidth=1)

    #     plt.xlabel("X1")
    #     plt.ylabel("X2")
    #     plt.show()

    import numpy as np
    import torch

    from sklearn.datasets import make_moons
    import torch

    def generate_moons_data(N, noise=0.1, random_seed=None):
        X, y = make_moons(n_samples=N, noise=noise, random_state=random_seed)
        X += np.array([0.0, 10.0])
        y = 2 * y - 1  # Convert labels from {0, 1} to {-1, 1}
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # Example usage
    N = 300
    X, y = generate_moons_data(N, noise=0.1, random_seed=42)
    y = (y + 1) // 2

    def generate_synthetic_data(N, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate N points for class 1
        x1_1 = np.random.normal(1, 0.5, (N // 2, 1))
        x2_1 = np.random.normal(1, 0.5, (N // 2, 1))
        X1 = np.hstack((x1_1, x2_1))
        y1 = np.zeros((N // 2,))

        # Generate N points for class -1
        x1_2 = np.random.normal(-0.5, 0.5, (N // 2, 1))
        x2_2 = np.random.normal(-0.5, 0.5, (N // 2, 1))
        X2 = np.hstack((x1_2, x2_2))
        y2 = np.ones((N // 2,))

        # Combine the two classes to get the final dataset
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def gen_three_clusters(n_samples):
        centers = [[0.0, 0.0], [0.5, 0.5], [0.5, 0.0]]
        cluster_stds = 0.05
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )
        return torch.tensor(data, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def gen_nine_clusters(n_samples):
        centers = [
            [0.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 1.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        cluster_stds = 0.1
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )
        return torch.tensor(data, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    X, y = gen_nine_clusters(N)
    # X, y = gen_three_clusters(N)
    # X, y = generate_synthetic_data(N, random_seed=42)
    y = y.long()

    # Initialize model and optimizer
    num_features = X.shape[1]
    num_fourier_features = 100  # The number of random Fourier features for approximation
    # model = PrimalLinearSVM(num_features=num_features, num_classes=9)
    model = PrimalPolySVM(num_features=num_features, degree=3, num_classes=9)
    # model = PrimalRBFSVM(num_features=num_features, num_fourier_features=num_fourier_features, gamma=4.0, num_classes=9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[100, 300, 500, 700, 900], gamma=0.5
    # )

    batch_size = 100

    # Training loop
    for epoch in range(1000):
        # Iterate batch-wise over the data
        for i in range(0, len(X), batch_size):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            # Compute scores and loss
            scores = model(X_batch)

            loss = model.loss(scores, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # scheduler.step()

        with torch.no_grad():
            scores = model(X)
            loss = model.loss(scores, y)
            preds = model.predict(X)
            acc = (preds == y).float().mean()
            print(f"Epoch: {epoch+1}/100, Loss={loss}, Accuracy={acc}")

    print(model.linear.weight)
    print(model.linear.bias)

    # Visualize decision boundary
    plot_decision_boundary(model, X.detach().numpy(), y.detach().numpy())
