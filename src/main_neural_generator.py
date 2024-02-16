#!/usr/bin/env python3
import shutil
import tarfile
import time
import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as tF
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.mlp import MLP

import matplotlib.animation as animation
from IPython.display import HTML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_gpu = torch.cuda.is_available()


# Moons
import sklearn.datasets

X, Y = sklearn.datasets.make_moons(1000, noise=0.3, random_state=0)

x_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(Y).long()

trainset = torch.utils.data.TensorDataset(x_train, y_train)

trainset, fisherset = torch.utils.data.random_split(trainset, [800, 200])

batch_size = 10
num_classes = 2
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=is_gpu, sampler=None
)
fisher_loader = torch.utils.data.DataLoader(
    fisherset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=is_gpu, sampler=None
)

from matplotlib.colors import ListedColormap

my_cmap = ListedColormap(sns.color_palette("Set2", 3).as_hex())
plt.scatter(X[:, 0], X[:, 1], s=50, c=Y, cmap=my_cmap)


# Train/Val Code
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Evaluates a model's top k accuracy

    Parameters:
        output (torch.autograd.Variable): model output
        target (torch.autograd.Variable): ground-truths/labels
        topk (list): list of integers specifying top-k precisions
            to be computed

    Returns:
        float: percentage of correct predictions
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(
    train_loader,
    model,
    num_classes,
    criterion,
    optimizer,
    epoch,
    device,
    print_freq=100,
    soft_labels=False,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inp, target) in enumerate(train_loader):
        inp, target = inp.to(device), target.to(device)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(inp)

        loss = criterion(output, target)

        # measure accuracy and record loss
        if soft_labels:
            # extract correct label for accuracy calculation
            # in the soft label case
            target = torch.argmax(target, dim=1)
        prec1 = accuracy(output, target)[0]

        losses.update(loss.item(), inp.size(0))
        top1.update(prec1.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )

    print(" * Train: Prec@1 {top1.avg:.3f}".format(top1=top1))


def validate(val_loader, model, criterion, epoch, device, print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inp, target) in enumerate(val_loader):
            inp, target = inp.to(device), target.to(device)

            # compute output
            output = model(inp)

            # compute loss
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)[0]

            losses.update(loss.item(), inp.size(0))
            top1.update(prec1.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                    )
                )

    print(" * Validation: Prec@1 {top1.avg:.3f}".format(top1=top1))

    return top1.avg


import torch.nn as nn
from torch.nn import init


sns.set()
sns.set_style("whitegrid")


def predict(model, x, device):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    x = x.to(device)
    with torch.no_grad():
        pred = tF.softmax(model.forward(x), dim=1)
    # take only probabilities of one class, as it is a binary problem
    probs = pred[:, 0].cpu()

    return probs


# adapted from https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
def plot_decision_boundary(pred_func, X, Y, x_min=None, x_max=None, y_min=None, y_max=None, title=None):
    # Set min and max values if None and give it some padding
    if x_min is None:
        x_min = X[:, 0].min() - 0.5
    if x_max is None:
        x_max = X[:, 0].max() + 0.5
    if y_min is None:
        y_min = X[:, 1].min() - 0.5
    if y_max is None:
        y_max = X[:, 1].max() + 0.5
    res = 200

    # x_min = -1.75
    # x_max = 2.75
    # y_min = -1.25
    # y_max = 1.75
    # generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    # predict on the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot the countour
    levels = np.linspace(0.0, 1.0, 11)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    axp = ax.contourf(xx, yy, Z, cmap="PRGn", levels=levels)
    c = plt.colorbar(axp, ax=[ax], location="left")
    c.ax.tick_params(labelsize=16)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, cmap="Paired", s=50)  # cmap=plt.cm.binary)
    plt.xticks(fontsize=18)
    ax.yaxis.tick_right()
    plt.yticks(fontsize=18)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))

    c.remove()
    if title:
        plt.title(title)
    plt.show()


#############
# Optimize  #
#############
sns.set()


def load_mlp(teacher_path) -> MLP:
    state_dict = torch.load(teacher_path, map_location="cpu")["state_dict"]
    for key in list(state_dict.keys()):
        if "model." in key:
            state_dict[key.replace("model.", "")] = state_dict.pop(key)
    model = MLP(
        in_features=2,
        num_classes=2,
        hidden_size=20,
        num_hidden=2,
    )
    model.load_state_dict(state_dict)
    return model


model = load_mlp(os.getenv("HOME") + "/results/CAKE-v6/synth-2-moons/default-teacher/0/teacher.ckpt").to(device)

print("Teacher MLP")
print(model)


def save_samples(
    batch_x,
    batch_y,
    index_set,
    samples_dir,
    noise,
    tar_archive: tarfile.TarFile,
):
    """
    Save the sampled batch to the disk at the given samples_dir.
    Args:
        batch_x: Batch data.
        batch_y: Batch labels.
        index_set: Current batch index.
        samples_dir: Directory where samples should be stored.
    """
    eps_dir_name = f"{index_set:0>4}_{noise:0>2.8f}"
    dirname = os.path.join(samples_dir, eps_dir_name)
    os.makedirs(dirname, exist_ok=True)
    for i in range(batch_x.shape[0]):
        # Construct global count
        global_count = index_set * batch_x.shape[0] + i

        data = batch_x[i]
        label = batch_y[i]

        filename = f"{global_count:0>9}.npz"
        path = os.path.join(dirname, filename)
        np.savez(path, data=data.numpy(), label=label.numpy())

        # Add file to tar archive
        tar_archive.add(path, arcname=eps_dir_name + "/" + filename)


#################################
# Generative Model Architecture #
#################################
class GenMLP(nn.Module):
    def __init__(self, hidden_size=10, z_dim=2, n_classes=2, device="cuda"):
        super(GenMLP, self).__init__()
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

    def forward(self):
        # sample one common z
        z = self.sample_z()

        # sample some conditional labels
        c1, c2 = self.cond_pair_sample()

        # if self.cond_pair_sample gets turned into a loop, so will this
        # lazy
        # decode step 1
        x1 = self.decode(z, c1)

        # decode step 2 with same z but different c
        x2 = self.decode(z, c2)

        # return the labels only, not the one hot vector as that's what most
        # pytorch environments expect
        return x1, x2, torch.argmax(c1, dim=1), torch.argmax(c2, dim=1)


######################
# Gen model opt code #
######################
def contrastive_loss(t_out1, t_out2):
    MSE = nn.MSELoss()
    contr_loss = MSE(t_out1, t_out2)
    return contr_loss


def gen_optimize(
    gen_model,
    teacher_model,
    contr_criterion,
    class_criterion,
    optimizer,
    num_steps,
    device,
    print_freq=100,
):
    losses = AverageMeter()
    top1 = AverageMeter()
    contrastive_loss = AverageMeter()

    # switch to train mode for generator and eval for teacher
    gen_model.train()
    teacher_model.eval()

    end = time.time()
    for step in range(num_steps):
        # sample pairs from gen model
        g_out1, g_out2, c_out1, c_out2 = gen_model()

        # compute teacher model output for x
        t_out1 = model(g_out1)
        t_out2 = model(g_out2)

        # contr_loss = contr_criterion(t_out1, t_out2)
        c1_loss = class_criterion(t_out1, c_out1)
        c2_loss = class_criterion(t_out2, c_out2)

        # overall loss
        # TODO: hardcoded alpha to 1, not sure if this should even be a hyperparam
        alpha = 0
        loss = c1_loss + c2_loss  # + alpha * contr_loss

        # measure accuracy and record loss
        prec1 = accuracy(t_out1, c_out1)[0]
        prec2 = accuracy(t_out2, c_out2)[0]

        losses.update(loss.item(), t_out1.size(0))
        # contrastive_loss.update(contr_loss.item(), t_out1.size(0))
        top1.update(prec1.item(), t_out1.size(0))
        top1.update(prec2.item(), t_out2.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_freq == 0:
            print(
                "Step: [{0}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Contrastive loss {contr_loss.val:.4f} ({contr_loss.avg:.4f})\t"
                "Acc {top1.val:.3f} ({top1.avg:.3f})".format(step, contr_loss=contrastive_loss, loss=losses, top1=top1)
            )


############
# Optimize #
############
gen_model = GenMLP(z_dim=5, hidden_size=10).to(device)

learning_rate = 1e-3
num_steps = 15000  # set to low value if someone runs entire notebook

optimizer = torch.optim.Adam(gen_model.parameters(), learning_rate)
class_criterion = nn.CrossEntropyLoss().to(device)

gen_optimize(
    gen_model,
    model,
    contrastive_loss,
    class_criterion,
    optimizer,
    num_steps,
    device,
    print_freq=500,
)

###############
# Vis samples #
###############
# generate a dataset
data = []
labels = []
for i in range(10):
    samples1_z1, samples1_z2, labels1, labels2 = gen_model()
    data.append(samples1_z1)
    labels.append(labels1)
    data.append(samples1_z2)
    labels.append(labels2)

# stack into one big tensor
gen_X = torch.concat(data, dim=0).detach().cpu()
gen_Y = torch.concat(labels, dim=0).detach().cpu()


# Open tar archive to serialize batches
run_dir = os.path.join(os.getenv("HOME"), "results", "CAKE-v6", "synth-2-moons", "gen-contrastive")
shutil.rmtree(run_dir)
samples_dir = os.path.join(run_dir, "samples")
os.makedirs(samples_dir, exist_ok=True)
tar_archive = tarfile.open(os.path.join(run_dir, "samples.tar"), "w")
# save_samples(
#     batch_x=gen_X,
#     batch_y=gen_Y,
#     index_set=0,
#     samples_dir=samples_dir,
#     noise=0.0,
#     tar_archive=tar_archive,
# )
tar_archive.close()

plot_decision_boundary(lambda x: predict(model, x, device), gen_X.numpy(), gen_Y.numpy())


# ################################
# # Contrastive + Repell version #
# ################################
# class CR_GenMLP(nn.Module):
#     def __init__(self, hidden_size=10, z_dim=2, n_classes=2, device="cuda"):
#         super(CR_GenMLP, self).__init__()
#         self.hidden_size = hidden_size
#         self.z_dim = z_dim
#         self.n_classes = n_classes
#         self.device = device
#         self.layers = nn.Sequential(
#             nn.Linear(z_dim + self.n_classes, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, n_classes),
#         )

#     def sample_z(self, batch_size=64):
#         # z ~ N(0,I)
#         # z = torch.randn(batch_size, self.z_dim).to(self.device)
#         # z ~ U[0,1)
#         z = torch.rand(batch_size, self.z_dim).to(self.device)
#         return z

#     def cond_pair_sample(self, batch_size=64):
#         target = torch.randint(high=self.n_classes, size=(1, batch_size))
#         # make one hot

#         y = [
#             sorted(list(range(self.n_classes)), key=lambda k: random.random())
#             for _ in range(batch_size)
#         ]
#         y = torch.tensor(y)

#         # we can turn this into a future loop for more than a pair of samples
#         c1 = torch.zeros(batch_size, self.n_classes)
#         c1[range(c1.shape[0]), y[:, 0].long()] = 1
#         c1 = c1.to(self.device)

#         c2 = torch.zeros(batch_size, self.n_classes)
#         c2[range(c2.shape[0]), y[:, 1].long()] = 1
#         c2 = c2.to(self.device)

#         return c1, c2

#     def decode(self, z, c):
#         x = torch.cat((z, c), dim=-1)
#         x = self.layers(x)
#         return x

#     def forward(self):
#         # sample one common z
#         z1 = self.sample_z()

#         # sample some conditional labels
#         c1, c2 = self.cond_pair_sample()

#         # if self.cond_pair_sample gets turned into a loop, so will this
#         # lazy
#         # decode step 1
#         x1_z1 = self.decode(z1, c1)

#         # decode step 2 with same z but different c
#         x2_z1 = self.decode(z1, c2)

#         # sample a new z for the loss term that repels intra class examples

#         # sample one common z
#         z2 = self.sample_z()

#         # if self.cond_pair_sample gets turned into a loop, so will this
#         # lazy
#         # decode step 1
#         x1_z2 = self.decode(z2, c1)

#         # decode step 2 with same z but different c
#         x2_z2 = self.decode(z2, c2)

#         # return the labels only, not the one hot vector as that's what most
#         # pytorch environments expect
#         return x1_z1, x2_z1, x1_z2, x2_z2, torch.argmax(c1, dim=1), torch.argmax(c2, dim=1)


############
# Opt code #
############
# def contrastive_loss(t_out1, t_out2):
#     MSE = nn.MSELoss()
#     contr_loss = MSE(t_out1, t_out2)
#     return contr_loss


# def gen_optimize(
#     gen_model,
#     teacher_model,
#     contr_criterion,
#     class_criterion,
#     optimizer,
#     num_steps,
#     device,
#     contr_weight=1,
#     rep_weight=1,
#     print_freq=100,
# ):
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     contrastive_loss = AverageMeter()
#     repell_loss = AverageMeter()

#     # switch to train mode for generator and eval for teacher
#     gen_model.train()
#     teacher_model.eval()

#     for step in range(num_steps):
#         # sample pairs from gen model
#         g_out1_z1, g_out2_z1, g_out1_z2, g_out2_z2, c_out1, c_out2 = gen_model()

#         # compute teacher model output for x
#         t_out1_z1 = model(g_out1_z1)
#         t_out2_z1 = model(g_out2_z1)
#         t_out1_z2 = model(g_out1_z2)
#         t_out2_z2 = model(g_out2_z2)

#         # intra-class contrastive between the same z and different class
#         contr_loss = contr_criterion(t_out1_z1, t_out2_z1) + contr_criterion(t_out1_z2, t_out2_z2)

#         # inter-class repell loss between different z and same class
#         # rep_loss = (1 / contr_criterion(t_out1_z1, t_out1_z2)) + (1 / contr_criterion(t_out2_z1, t_out2_z2))
#         rep_loss = (1 / contr_criterion(g_out1_z1, g_out1_z2)) + (
#             1 / contr_criterion(g_out2_z1, g_out2_z2)
#         )

#         c1_loss = class_criterion(t_out1_z1, c_out1) + class_criterion(t_out1_z2, c_out1)
#         c2_loss = class_criterion(t_out2_z1, c_out2) + class_criterion(t_out2_z2, c_out2)

#         # overall loss
#         rep_weight = 0
#         loss = 10 * (c1_loss + c2_loss) + contr_weight * contr_loss + rep_weight * rep_loss

#         # measure accuracy and record loss
#         prec1 = accuracy(t_out1_z1, c_out1)[0]
#         prec2 = accuracy(t_out2_z1, c_out2)[0]
#         prec3 = accuracy(t_out1_z2, c_out1)[0]
#         prec4 = accuracy(t_out2_z2, c_out2)[0]

#         losses.update(loss.item(), t_out1_z1.size(0))
#         contrastive_loss.update(contr_weight * contr_loss.item(), t_out1_z1.size(0))
#         repell_loss.update(rep_weight * rep_loss.item(), t_out1_z2.size(0))

#         top1.update(prec1.item(), t_out1_z1.size(0))
#         top1.update(prec2.item(), t_out2_z1.size(0))
#         top1.update(prec3.item(), t_out1_z2.size(0))
#         top1.update(prec4.item(), t_out2_z2.size(0))

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if step % print_freq == 0:
#             print(
#                 "Step: [{0}]\t"
#                 "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
#                 "Contrastive loss {contr_loss.val:.4f} ({contr_loss.avg:.4f})\t"
#                 "Repell loss {repell_loss.val:.4f} ({repell_loss.avg:.4f})\t"
#                 "Acc {top1.val:.3f} ({top1.avg:.3f})".format(
#                     step,
#                     i,
#                     len(train_loader),
#                     contr_loss=contrastive_loss,
#                     repell_loss=repell_loss,
#                     loss=losses,
#                     top1=top1,
#                 )
#             )


############
# Opt code #
############
# gen_model = CR_GenMLP(z_dim=10, hidden_size=100).to(device)

# learning_rate = 1e-3
# num_steps = 5000  # set to low value if someone runs entire notebook

# optimizer = torch.optim.Adam(gen_model.parameters(), learning_rate)
# class_criterion = nn.CrossEntropyLoss().to(device)

# gen_optimize(
#     gen_model,
#     model,
#     contrastive_loss,
#     class_criterion,
#     optimizer,
#     num_steps,
#     device,
#     contr_weight=1,
#     rep_weight=0.001,
#     print_freq=500,
# )

# #############
# # Visualize #
# #############
# # generate a dataset
# data = []
# labels = []
# for i in range(10):
#     samples1_z1, samples2_z1, samples1_z2, samples2_z2, labels1, labels2 = gen_model()
#     data.append(samples1_z1)
#     data.append(samples1_z2)
#     labels.append(labels1)
#     labels.append(labels1)
#     data.append(samples2_z2)
#     data.append(samples2_z2)
#     labels.append(labels2)
#     labels.append(labels2)

# # stack into one big tensor
# gen_X = torch.concat(data, dim=0).detach().cpu()
# gen_Y = torch.concat(labels, dim=0).detach().cpu()


# # Open tar archive to serialize batches
# run_dir = os.path.join(os.getenv("HOME"), "results", "CAKE-v6", "synth-2-moons", "gen-contrastive-repell")
# shutil.rmtree(run_dir)
# samples_dir = os.path.join(run_dir, "samples")
# os.makedirs(samples_dir, exist_ok=True)
# tar_archive = tarfile.open(os.path.join(run_dir, "samples.tar"), "w")
# save_samples(
#     batch_x=gen_X,
#     batch_y=gen_Y,
#     index_set=0,
#     samples_dir=samples_dir,
#     noise=0.0,
#     tar_archive=tar_archive,
# )
# tar_archive.close()

# plot_decision_boundary(lambda x: predict(model, x, device), gen_X.numpy(), gen_Y.numpy())
