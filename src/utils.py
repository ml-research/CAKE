import io
import contextlib
import datetime
import errno
import io
import logging
import os
import pathlib
import random
import shutil
import time
import traceback
from typing import List, Tuple, Union

import PIL
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torchvision.utils
import wandb
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.loggers import WandbLogger
from torch.backends import cudnn as cudnn
from torchvision.transforms import ToTensor

from data import generate_data


def samples_path(run_dir: str):
    samples_path = os.path.join(run_dir, "samples")
    os.makedirs(samples_path, exist_ok=True)
    return samples_path


def _checkpoint_path(run_dir: str, model_name: str) -> str:
    """
    Constructs the path to a model checkpoint.

    Args:
        run_dir: The run directory.

    Returns:
        str: The path to the model checkpoint.
    """
    return os.path.join(run_dir, f"{model_name}.ckpt")


def student_checkpoint_path(run_dir: str) -> str:
    """
    Constructs the path to the teacher model checkpoint.

    Args:
        run_dir: The run directory.

    Returns:
        str: The path to the model checkpoint.
    """
    return _checkpoint_path(run_dir, "student")


def teacher_checkpoint_path(run_dir: str) -> str:
    """
    Constructs the path to the student model checkpoint.

    Args:
        run_dir: The run directory.

    Returns:
        str: The path to the model checkpoint.
    """
    return _checkpoint_path(run_dir, "teacher")


def count_params(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_project_name() -> str:
    """
    Get the project name.

    Returns:
        str: Project name.
    """
    return open("./PROJECT_NAME").readlines()[0].strip()


logger = logging.getLogger(get_project_name())


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big")
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def detect_anomaly(losses: torch.Tensor, iteration: int):
    """Check if loss is finite .

    Args:
        losses (torch.Tensor): Loss to be checked.
        iteration (int): Current iteration.

    Raises:
        FloatingPointError: Loss was not finite.
    """
    # use a new stream so the ops don't wait for DDP
    with (
        torch.cuda.stream(torch.cuda.Stream(device=losses.device))
        if losses.device.type == "cuda"
        else contextlib.nullcontext()
    ):
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Loss became infinite or NaN at iteration={}!".format(iteration))


def catch_exception(output_directory: str, e: Exception):
    """Catch exception and rename output directory.

    Args:
        save_path (str): Model output directory.
        e (Exception): Exception which was catched.

    Raises:
        Exception: Exception which was catched.
    """
    # Log error message
    tbstr = "".join(traceback.extract_tb(e.__traceback__).format())
    errormsg = f"Traceback:\n{tbstr}\nError: {e}"

    # Rename output dir
    src = output_directory
    if src.endswith("/"):
        src = src[:-1]
    dst = src + "_error"

    # Write error to separate file
    with open(os.path.join(output_directory, "error.txt"), "w") as f:
        f.write(errormsg)

    logger.error("Error caught!")
    logger.error(f"Moving output directory from")
    logger.error(src)
    logger.error("to")
    logger.error(dst)

    shutil.move(src, dst)
    raise e


def catch_kb_interrupt(output_directory):
    """Catch keyboard interrupt and rename output directory.

    Args:
        output_directory (str): Output directory.
    """
    # Rename output dir
    src = output_directory
    if src.endswith("/"):
        src = src[:-1]
    dst = src + "_interrupted"

    logger.error(f"Keyboard interruption catched.")
    logger.error(f"Moving output directory from")
    logger.error(src)
    logger.error("to")
    logger.error(dst)

    shutil.move(src, dst)


def xor(a: bool, b: bool) -> bool:
    """Perform the XOR operation between a and b."""
    return (a and not b) or (not a and b)


def plot_tensor(x: torch.Tensor):
    plt.figure()
    if x.dim() == 4:
        x = torchvision.utils.make_grid(x)
    plt.imshow(x.permute(1, 2, 0).detach().cpu().numpy())
    plt.show()
    plt.close()


def load_from_checkpoint(run_dir, load_fn, args):
    """Loads the model from a checkpoint.

    Args:
        load_fn: The function to load the model from a checkpoint.
    Returns:
        The loaded model.
    """
    ckpt_dir = os.path.join(run_dir, "tb", "version_0", "checkpoints")
    files = os.listdir(ckpt_dir)
    assert len(files) > 0, "Checkpoint directory is empty"
    ckpt_path = os.path.join(ckpt_dir, files[-1])
    model = load_fn(checkpoint_path=ckpt_path, args=args)
    return model


def save_samples(generate_samples, samples_dir, num_samples, nrow):
    for i in range(5):
        samples = generate_samples(num_samples)
        grid = torchvision.utils.make_grid(samples, nrow=nrow, pad_value=0.0, normalize=True)
        torchvision.utils.save_image(grid, os.path.join(samples_dir, f"{i}.png"))


from matplotlib.cm import tab10

TEXTWIDTH = 5.78853
LINEWIDTH = 0.75
ARROW_HEADWIDTH = 5
colors = tab10.colors


def get_figsize(scale: float, aspect_ratio=0.8) -> Tuple[float, float]:
    """
    Scale the default figure size to: (scale * TEXTWIDTH, scale * aspect_ratio * TEXTWIDTH).

    Args:
      scale(float): Figsize scale. Should be lower than 1.0.
      aspect_ratio(float): Aspect ratio (as scale), height to width. (Default value = 0.8)

    Returns:
      Tuple: Tuple containing (width, height) of the figure.

    """
    height = aspect_ratio * TEXTWIDTH
    widht = TEXTWIDTH
    return (scale * widht, scale * height)


def set_style():
    matplotlib.use("pgf")
    plt.style.use(["science", "grid"])  # Need SciencePlots pip package
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )


def plot_distribution(model, dataset_name, logger_wandb: WandbLogger = None):
    with torch.no_grad():
        data, teachers = generate_data(dataset_name, n_samples=1000)
        fig = plt.figure(figsize=get_figsize(1.0))
        data_cpu = data.cpu()
        delta = 0.05
        xmin, xmax = data_cpu[:, 0].min(), data_cpu[:, 0].max()
        ymin, ymax = data_cpu[:, 1].min(), data_cpu[:, 1].max()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        x = np.arange(xmin * 1.05, xmax * 1.05, delta)
        y = np.arange(ymin * 1.05, ymax * 1.05, delta)
        X, Y = np.meshgrid(x, y)

        Z = torch.exp(model(torch.from_numpy(np.c_[X.flatten(), Y.flatten()]).to(data.device).float()).float()).cpu()
        Z = Z.view(X.shape)
        CS = plt.contourf(X, Y, Z, 100, cmap=plt.cm.viridis)
        plt.colorbar(CS)

        plt.scatter(
            *data_cpu[:500].T,
            label="Data",
            ec="black",
            lw=0.5,
            s=10,
            alpha=0.5,
            color=colors[1],
        )

        plt.xlabel("$X_0$")
        plt.ylabel("$X_1$")
        plt.title(f"Learned PDF")

        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)

        # Add figure in numpy "image" to TensorBoard writer
        logger_wandb.log_image("distribution", images=[image])
        plt.close(fig)


def preprocess_cfg(cfg: DictConfig):
    """
    Preprocesses the config file.
    Replace defaults if not set (such as data/results dir).

    Args:
        cfg: Config file.
    """
    home = os.getenv("HOME")

    # If results dir is not set, get from ENV, else take ~/data
    if "data_dir" not in cfg.env:
        cfg.env.data_dir = os.getenv("DATA_DIR", os.path.join(home, "data"))

    # If results dir is not set, get from ENV, else take ~/results
    if "results_dir" not in cfg.env:
        cfg.env.results_dir = os.getenv("RESULTS_DIR", os.path.join(home, "results"))

    # If FP16/FP32 is given, convert to int (else it's "bf16", keep string)
    if cfg.env.precision == "16" or cfg.env.precision == "32":
        cfg.env.precision = int(cfg.env.precision)

    if "profiler" not in cfg.env:
        cfg.env.profiler = None  # Accepted by PyTorch Lightning Trainer class

    if "tag" not in cfg.env:
        cfg.env.tag = None

    if "group_tag" not in cfg.env:
        cfg.env.group_tag = None

    if "seed" not in cfg.env:
        cfg.env.seed = int(time.time())

    if "notes" not in cfg.env:
        cfg.env.notes = None

    # Take the min between the chosen num_workers and the number of CPUs
    cfg.data.num_workers = min(cfg.data.num_workers, os.cpu_count())


def get_device_type(cfg: DictConfig) -> str:
    if torch.backends.mps.is_available():  # Mac
        device_type = "cpu"
    elif torch.cuda.is_available():  # CUDA
        device_type = "cuda"
    else:  # CPU
        device_type = "cpu"
    return device_type


def get_lightning_accelerator_and_devices(cfg: DictConfig) -> Tuple[str, Union[str, List]]:
    # Setup devices
    if torch.backends.mps.is_available():  # Mac
        # accelerator = "mps"
        # devices = 1
        accelerator = "cpu"
        devices = 1
    elif torch.cuda.is_available():  # CUDA
        accelerator = "gpu"
        devices = [int(cfg.env.gpu)]
    else:  # CPU
        accelerator = "cpu"
        devices = 1
    return accelerator, devices


def make_wandb_logger(cfg: DictConfig, run_dir: str):
    # Add config to wandb
    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Necessary when running multiple runs in parallel
    reinit = True

    logger_wandb = WandbLogger(
        name=cfg.env.tag,
        project=cfg.env.project_name,
        group=cfg.env.group_tag,
        offline=not cfg.env.wandb,
        notes=cfg.env.notes,
        config=cfg_container,
        reinit=reinit,
        settings=wandb.Settings(start_method="thread"),
        save_dir=run_dir,
    )
    return logger_wandb


def get_devices(cfg: DictConfig) -> List[int]:
    if type(cfg.env.gpu) == int:
        gpus = [cfg.env.gpu]
    elif type(cfg.env.gpu) == ListConfig:
        gpus = list(cfg.env.gpu)
    else:
        raise ValueError("Invalid env.gpu argument.")

    return gpus
