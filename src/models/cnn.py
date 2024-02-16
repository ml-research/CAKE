from torch import nn
from torch.nn import functional as F


class CnnWithAdaptivePool(nn.Module):
    """
    A convolutional neural network with adaptive pooling.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        last_layer_dim (int): Dimension of the last layer.
    """

    def __init__(self, in_channels: int, num_classes: int, last_layer_dim: int):
        super().__init__()
        self.last_layer_dim = last_layer_dim
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 1)
        self.conv3 = nn.Conv2d(64, last_layer_dim, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(last_layer_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, self.last_layer_dim)
        x = self.fc1(x)
        return x


class Cnn(nn.Module):
    """
    Convolutional Neural Network model for image classification.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(1 * 1 * 64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 1 * 1 * 64)
        x = self.fc1(x)
        return x


class CnnBig(nn.Module):
    """
    A convolutional neural network with four convolutional layers and one fully connected layer.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 128, 3, 1)
        self.fc1 = nn.Linear(1 * 1 * 128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 1 * 1 * 128)
        x = self.fc1(x)
        return x


class CnnTiny(nn.Module):
    """
    A small convolutional neural network for image classification.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 4, 1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1)
        self.fc1 = nn.Linear(1 * 1 * 16, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 1 * 1 * 16)
        x = self.fc1(x)
        return x
