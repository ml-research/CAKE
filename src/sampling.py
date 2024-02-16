import logging
import os
import random
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from rich.progress import Progress
from rtpt import RTPT
from torch.nn import functional as F

from src.data import Shape, get_number_of_classes
from src.models.lit import DiscriminativeModel
from src.tardataset import TarDataset
from src.utils import get_device_type

# A logger for this file
logger = logging.getLogger(__name__)


def check_finite(loss, loss_dict):
    """
    Check if the loss is finite. If not, log an error message and raise a RuntimeError.

    Args:
        loss (torch.Tensor): The loss tensor to check.
        loss_dict (dict): A dictionary containing additional information about the loss.

    Raises:
        RuntimeError: If the loss is not finite.
    """
    # Stop if loss becomes nan or inf
    if not torch.isfinite(loss).all():
        msg = f"Loss was not finite:\n{loss_dict}"
        logger.error(msg)
        raise RuntimeError(msg)


def add_noise_langevin(x: torch.Tensor, eps: float):
    """Add noise to the sample, according to langevin dynamics."""
    return x + np.sqrt(2 * eps) * torch.randn_like(x)


def kl_loss(mu, var, mu_prior=0.0, var_prior=1.0):
    """
    Compute KL divergence between a Gaussian distributions N(mu, var) and N(0, 1).

    Args:
        mu: mean of the distribution
        var: variance of the distribution

    Returns:
        KL divergence

    """
    return torch.sum(kl_div(mu, var, mu_prior, var_prior))


def kl_div(mu_1, var_1, mu_2, var_2):
    """
    Compute KL divergence between two Gaussian distributions N(mu_1, var_1) and N(mu_2, var_2).
    Args:
        mu_1: mean of the distribution 1
        var_1: variance of the distribution 1
        mu_2: mean of the distribution 2
        var_2: variance of the distribution 2

    Returns:
        KL divergence
    """
    return torch.log(var_2 / var_1) + (var_1**2 + (mu_1 - mu_2) ** 2) / (2 * var_2**2) - 0.5


def total_variation(img: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    r"""Function that computes Total Variation according to [1].

    Modified code from https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html

    Args:
        img: the input image with shape :math:`(*, H, W)`.
        reduction : Specifies the reduction to apply to the output: ``'mean'`` | ``'sum'``.
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.

    Return:
         a tensor with shape :math:`(*,)`.

    Examples:
        >>> total_variation(torch.ones(4, 4))
        tensor(0.)
        >>> total_variation(torch.ones(2, 5, 3, 4, 4)).shape
        torch.Size([2, 5, 3])

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       total_variation_denoising.html>`__.
       Total Variation is formulated with summation, however this is not resolution invariant.
       Thus, `reduction='mean'` was added as an optional reduction method.

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """

    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    res1 = pixel_dif1.abs()
    res2 = pixel_dif2.abs()

    reduce_axes = (-2, -1)
    if reduction == "mean":
        if img.is_floating_point():
            res1 = res1.to(img).mean(dim=reduce_axes)
            res2 = res2.to(img).mean(dim=reduce_axes)
        else:
            res1 = res1.float().mean(dim=reduce_axes)
            res2 = res2.float().mean(dim=reduce_axes)
    elif reduction == "sum":
        res1 = res1.sum(dim=reduce_axes)
        res2 = res2.sum(dim=reduce_axes)

    return res1 + res2


def generate_samples(
    model_teacher: DiscriminativeModel,
    shape: Shape,
    cfg: DictConfig,
    device,
    logger_wandb: WandbLogger,
    samples_dir: str,
):
    """
    Generates synthetic samples from a teacher model.

    Args:
        model_teacher (DiscriminativeModel): The teacher model to use for generating samples.
        shape (Shape): The shape of the input data.
        cfg (DictConfig): The configuration dictionary.
        device: The device to use for generating samples.
        logger_wandb (WandbLogger): The logger to use for logging samples.
        samples_dir (str): The directory to save the generated samples.

    Returns:
        TarDataset: A TarDataset containing the generated samples.
    """
    # Get number of classes for this specific dataset
    num_classes = get_number_of_classes(cfg.data.dataset)

    # Precision
    precision = cfg.env.precision
    if precision == "16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Get sampling hyper-parameters
    num_steps = cfg.sampling.num_steps
    noise_eps_start = cfg.sampling.noise
    batch_size = cfg.sampling.batch_size
    num_batches = cfg.sampling.num_batches
    num_groups = cfg.sampling.num_groups

    assert (
        batch_size % num_groups == 0
    ), f"Batch size must be divisble by number of groups to ensure that each group has an equal number of datapoints (batch_size={batch_size}, num_groups={num_groups})"
    group_batch_size = batch_size // num_groups

    # Get optimizer config
    optimizer_type = cfg.sampling.optim.type

    # Setup RTPT (updates process name with progress)
    rtpt = RTPT(
        name_initials="SB",
        experiment_name=f"CAKE_sampling" + ("_" + str(cfg.env.tag) if cfg.env.tag else ""),
        max_iterations=num_batches,
    )

    # Ensure that the model is on the correct device and the teacher is in eval mode
    model_teacher.to(device)
    model_teacher.eval()

    # Disable gradients for teacher
    for param in model_teacher.parameters():
        param.requires_grad_(False)

    assert num_groups <= num_classes, f"num_groups ({num_groups}) was larger than num_classes (" f"{num_classes})"
    # Get run_dir
    run_dir = Path(samples_dir).parent

    # Open tar archive to serialize batches
    tar_archive_path = os.path.join(run_dir, "samples.tar")
    if os.path.exists(tar_archive_path):
        logger.info(f"Removing existing tar archive at {tar_archive_path}")
        os.remove(tar_archive_path)
        logger.info("Done ...")
    tar_archive = tarfile.open(tar_archive_path, "w")

    # Iterate over number of sets to generate
    rtpt.start()
    rich_auto_refresh = not cfg.env.debug  # Disable rich autorefresh when debugging
    logger.info("Starting sampling ...")
    with Progress(auto_refresh=rich_auto_refresh) as progress:
        task_sampling = progress.add_task("Batches", total=num_batches)
        for index_set in range(num_batches):
            # Stop early for debugging
            if index_set > 1 and cfg.env.debug:
                break

            # Apply noise decay if chosen
            if cfg.sampling.noise_decay:
                eps = interpolate(
                    t=index_set,
                    T=num_batches,
                    start_value=noise_eps_start,
                    decay_magnitude=cfg.sampling.noise_decay_magnitude,
                    schedule=cfg.sampling.noise_decay_schedule,
                )
            else:
                eps = noise_eps_start

            # Initialize some random batch
            batch_x = torch.randn(
                group_batch_size,
                num_groups,
                *shape,
                device=device,
                dtype=dtype,
            )
            batch_x.requires_grad_(True)

            # Sample random labels by drawing without putting back batch
            y = [torch.randperm(num_classes, device=device) for _ in range(group_batch_size)]
            y = torch.stack(y, dim=0)

            batch_y = y[:, :num_groups]

            # Collect parameters for which we need to track the gradients
            params = [batch_x]

            if optimizer_type == "sgd":
                optimizer = torch.optim.SGD(lr=eps, params=params)
            elif optimizer_type == "adam":
                optimizer = torch.optim.Adam(lr=eps, params=params)
            else:
                raise ValueError(f"Invalid optimizer_type: {optimizer_type}")

            device_type = get_device_type(cfg)

            # Iterate over number of steps
            for step in range(num_steps):
                # Stop early for debugging
                if step > 1 and cfg.env.debug:
                    break

                with torch.autocast(device_type=device_type):
                    # Make predictions
                    preds = model_teacher(batch_x.view(batch_size, *shape))
                    preds = preds.view(group_batch_size, num_groups, num_classes)
                    loss_dict = compute_loss(
                        preds=preds,
                        batch_y=batch_y,
                        cfg=cfg,
                        batch_x=batch_x,
                        num_classes=num_classes,
                        num_groups=num_groups,
                    )

                    # Weight losses (Note: this requires cfg.sampling.weight to have the same keys as the dict returned by compute_loss)
                    loss_dict_weighted = {k: v * cfg.sampling.weight[k] for k, v in loss_dict.items()}

                    # Final loss is the sum of partial losses
                    loss = sum(loss_dict_weighted.values())

                optimizer.zero_grad()
                # Compute gradients
                loss.backward()

                # Perform a gradient step, updates batch_x (x <- x + grad_x(logp(x)))
                optimizer.step()

                # Add noise according to the langevin dynamics
                if cfg.sampling.langevin:
                    batch_x.data = batch_x.data + np.sqrt(2 * eps) * torch.randn_like(batch_x)

            # Check that loss is finite (raises on nan/inf)
            check_finite(loss, loss_dict)

            # Measure accuracy from last batch
            accuracy = (batch_y == preds.argmax(-1)).sum() / (batch_size) * 100

            # Collect metrics and accuracy (take the last loss_dict and accuracy)
            metrics = {"sampling/eps": eps, "sampling/accuracy": accuracy}
            for key, value in loss_dict_weighted.items():
                metrics["sampling/loss_" + key] = value

            # Log to wandb
            if index_set % cfg.env.log_every_n_steps == 0:
                logger_wandb.log_metrics(
                    metrics=metrics,
                    step=index_set,
                )

            # Update process subtitle
            rtpt.step(f"batch=[{index_set}|{num_batches}]")

            # Obtain model predictions for the generated batch
            preds = model_teacher(batch_x.view(batch_size, *shape))
            preds = preds.view(group_batch_size, num_groups, num_classes)

            # also now optionally replace the conditional label with the actual
            # distribution over output values of the teacher model: soft labels
            if cfg.sampling.smooth_labels:
                batch_y_view = F.softmax(preds, dim=2).view(batch_size, num_classes).detach()
            else:
                batch_y_view = preds.argmax(dim=2).view(batch_size).detach()

            # Reshape into correct batch size shape
            batch_x_view = batch_x.view(batch_size, *shape)

            # Serialize samples to disk
            save_sample_batch(
                batch_x_view.detach().cpu(),
                batch_y_view.detach().cpu(),
                index_set,
                samples_dir,
                eps,
                tar_archive,
            )

            # Print loss
            loss_str = " | ".join([f"{k}: [green]{v.item():.3f}[white]" for k, v in loss_dict.items()])
            descr = f"[{index_set + 1}/{num_batches}] Sampling - [ acc: [green]{accuracy:.2f}[white] | {loss_str} ]"
            progress.update(task_sampling, advance=1.0, description=descr)

    # Close tar archive
    tar_archive.close()

    # Remove all files
    shutil.rmtree(samples_dir, ignore_errors=True)

    run_dir = Path(samples_dir).parent
    return TarDataset(
        archive=os.path.join(run_dir, "samples.tar"),
        eps_min=cfg.student.data.eps_min,
        eps_max=cfg.student.data.eps_max,
    )


def interpolate(t: int, T: int, start_value: float, decay_magnitude: int, schedule: str, base=10):
    # Compute start
    start_magnitude = np.log(start_value) / np.log(base)

    if schedule == "linear":
        # linearly interpolate magnitude
        current_magnitude = start_magnitude - (t / T) * decay_magnitude
    elif schedule == "cosine":
        end_magnitude = start_magnitude - decay_magnitude
        current_magnitude = end_magnitude + 0.5 * decay_magnitude * (1 + np.cos((t / T) * np.pi))
    else:
        raise ValueError(f"Invlaid noise_decay_schedule {schedule}")

    return base**current_magnitude


def compute_loss(
    preds,
    batch_y,
    cfg,
    batch_x,
    num_classes,
    num_groups,
):
    """
    Compute the loss for the current batch.

    Args:
        preds: Predictions for the current batch.
        batch_y: Labels for the current batch.
        cfg: Config object.
        batch_x: Input batch.
        num_classes: Number of classes.
        num_groups: Number of groups.
        num_classes: Number of classes.

    Returns:
        loss: Total loss, classification loss, contrastive loss, and kl loss.

    """
    # Vectorized computation across all groups (merge num_samples and num_groups dim)
    preds_v = preds.view(-1, num_classes)

    # Classification loss
    # Vectorized computation across all groups (merge num_samples and num_groups dim)
    loss_classification = torch.nn.functional.cross_entropy(preds_v, batch_y.view(-1))

    # Scale by number of groups
    loss_classification *= num_groups

    # Contrastive loss
    # Vectorized version across all group combinations
    preds_A = preds.unsqueeze(1)
    preds_B = preds.unsqueeze(2)

    # Scale by 0.5 since every combination is accounted for twice
    # Scale by num_group**2 since MSE loss has reduction mean by default and now we also not
    # only average over batch but over batch*num_groups**2
    # TODO: this is computational overhead (factor x2), can we remove this?
    if cfg.sampling.weight.contr > 0.0:
        if cfg.sampling.contr_method == "mse":
            loss_contrastive = torch.nn.functional.mse_loss(preds_A, preds_B, reduction="mean")
        elif cfg.sampling.contr_method == "snn":
            loss_contrastive = 1 / soft_nn_loss(preds_v, batch_y.view(-1, 1), temperature=cfg.sampling.snn_temperature)
        else:
            raise ValueError(f"Invalid contrastive method: {cfg.sampling.contr_method}")
    else:
        loss_contrastive = torch.zeros(1, device=batch_x.device)

    loss_contrastive *= 0.5 * num_groups**2

    # Information entropy loss
    if cfg.sampling.weight.entropy > 0.0:
        p = torch.nn.functional.softmax(preds_v, dim=-1).mean(dim=0)
        loss_information_entropy = (p * torch.log10(p)).mean()
    else:
        loss_information_entropy = torch.zeros(1, device=batch_x.device)

    # Total variation loss
    if cfg.sampling.weight.tv > 0.0:
        loss_tv = torch.mean(total_variation(batch_x, reduction="mean"))
    else:
        loss_tv = torch.zeros(1, device=batch_x.device)

    return {
        "cls": loss_classification,
        "contr": loss_contrastive,
        "tv": loss_tv,
        "entropy": loss_information_entropy,
    }


def build_masks(labels, batch_size):
    labels = labels.view(-1, 1)
    mask = labels == labels.t()
    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    pos_mask = mask & ~diag_mask
    return pos_mask, ~mask


def distance(embeddings):
    pairwise_distances = torch.sum((embeddings.unsqueeze(1) - embeddings.unsqueeze(0)) ** 2, dim=2)
    return pairwise_distances


def soft_nn_loss(batch_x, batch_y, temperature):
    batch_size = batch_y.size(0)
    eps = 1e-9

    pairwise_dist = distance(batch_x) / temperature
    negexpd = torch.exp(-pairwise_dist)

    # Mask out diagonal entries
    diag_mask = torch.eye(batch_size, dtype=torch.float32, device=batch_x.device)
    negexpd = negexpd * (1 - diag_mask)

    # Creating mask to sample same class neighborhood
    pos_mask, _ = build_masks(batch_y, batch_size)
    pos_mask = pos_mask.float()

    # All class neighborhood
    alcn = torch.sum(negexpd, dim=1)

    # Same class neighborhood
    sacn = torch.sum(negexpd * pos_mask, dim=1)

    # Exclude examples with unique class from loss calculation
    excl = (torch.sum(pos_mask, dim=1) != 0).float()

    loss = sacn / (alcn + eps)
    loss = torch.log(eps + loss) * excl
    loss = -torch.mean(loss)
    return loss


def save_sample_batch(
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
        noise: Noise level used for sampling.
        tar_archive: Tar archive to store samples in.
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
