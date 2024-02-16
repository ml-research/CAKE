# adapted from https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
import io

import PIL
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from torch.nn import functional as F


def predict(model, x, device):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    x = x.to(device)
    with torch.no_grad():
        pred = F.softmax(model.forward(x), dim=1)
    # take only probabilities of one class, as it is a binary problem
    probs = pred[:, 0].cpu()
    return probs


def plot_data_and_decision_boundary(
    pred_func,
    data,
    labels,
    x_min=-2.0,
    x_max=2.0,
    y_min=-2.0,
    y_max=2.0,
    title=None,
    step=None,
    logger_wandb=None,
    log_key=None,
    vis_data_subset_size=500,
    device=torch.device("cpu"),
):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Select subset
    indices_subset = torch.randperm(data.shape[0])[:vis_data_subset_size]
    data = data[indices_subset]
    labels = labels[indices_subset]

    # Set min and max values if None and give it some padding
    if x_min is None:
        x_min = data[:, 0].min() - 0.1
    if x_max is None:
        x_max = data[:, 0].max() + 0.1
    if y_min is None:
        y_min = data[:, 1].min() - 0.1
    if y_max is None:
        y_max = data[:, 1].max() + 0.1
    res = 200
    # generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))

    # predict on the whole grid
    model_input = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
    with torch.no_grad():
        Z = pred_func(model_input)
        if Z.dim() == 2:
            Z = Z[:, 0]
    Z = Z.reshape(xx.shape).detach().cpu().numpy()

    # Normalize Z between 0 and 1
    Z = (Z - Z.min()) / (Z.max() - Z.min())

    # plot the countour
    levels = np.linspace(0.0, 1.0, 11)
    plt.contourf(xx, yy, Z, cmap="PRGn", levels=levels)
    c = plt.colorbar()
    if len(labels.shape) == 2:
        colors = np.argmax(labels, axis=1)
    else:
        colors = labels
    plt.scatter(data[:, 0], data[:, 1], marker="x", c=colors, cmap="Paired")  # cmap=plt.cm.binary)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    if title:
        plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    # Add figure in numpy "image" to TensorBoard writer
    logger_wandb.log_image(log_key, images=[image], step=step)

    plt.close()
