#!/usr/bin/env python
import os

import sys

from rich.traceback import install

install(show_locals=False)

import torch.utils.data
import torchvision.utils
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import (
    ModelSummary,
)
from torch.utils.data import Dataset, random_split


from src.data import (
    build_dataloader,
    generate_data,
    make_dataloader_from_dataset,
    get_data_shape,
    vis_student_synth_data,
)
from src.sampling import generate_samples
from src.models.lit import DiscriminativeModel, ModelType
from src.tardataset import TarDataset
from src.vis import plot_data_and_decision_boundary
from src.utils import (
    get_lightning_accelerator_and_devices,
    preprocess_cfg,
    samples_path,
    make_wandb_logger,
    get_devices,
    student_checkpoint_path,
)
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb

import logging

# A logger for this file
logger = logging.getLogger(__name__)

import warnings

# Ignore contrastive loss warning when broadcasting
warnings.filterwarnings("ignore", message="Using a target size .* that is different to the input size .*")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    preprocess_cfg(cfg)

    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir

    # Save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # Safe run_dir in config (use open_dict to make config writable)
    with open_dict(cfg):
        cfg.env.run_dir = run_dir

    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
    logger.info("Run dir: " + run_dir)

    seed_everything(cfg.env.seed, workers=True)

    # Create dataloader
    train_loader, val_loader, test_loader = build_dataloader(
        dataset_name=cfg.data.dataset,
        data_dir=cfg.env.data_dir,
        num_workers=cfg.data.num_workers,
        batch_size=cfg.teacher.bs,
        full_train_set=cfg.data.full_train_set,
    )

    # Create wandb logger
    logger_wandb = make_wandb_logger(cfg, run_dir=run_dir)

    if cfg.run.teacher and not cfg.reference_dir.teacher:
        # Construct new teacher model
        model_teacher = DiscriminativeModel(
            cfg=cfg,
            learning_rate=cfg.teacher.lr,
            epochs=cfg.teacher.epochs,
            model_cfg=cfg.teacher,
            weight_decay=cfg.teacher.weight_decay,
            steps_per_epoch=len(train_loader),
            model_type=ModelType.TEACHER,
        )

        train_teacher = True
    elif cfg.reference_dir.teacher:
        logger.info(f"Teacher reference dir set. Loading teacher model from " f"{cfg.reference_dir.teacher}")

        model_teacher_checkpoint_path = teacher_checkpoint_path(cfg.reference_dir.teacher)

        # Check if file exists
        if not os.path.isfile(model_teacher_checkpoint_path):
            logger.error(f"No teacher model checkpoint found at: {model_teacher_checkpoint_path}")
            raise FileNotFoundError(model_teacher_checkpoint_path)

        model_teacher = DiscriminativeModel.load_from_checkpoint(
            checkpoint_path=model_teacher_checkpoint_path,
            cfg=cfg,
            learning_rate=cfg.teacher.lr,
            epochs=cfg.teacher.epochs,
            model_cfg=cfg.teacher,
            weight_decay=cfg.teacher.weight_decay,
            steps_per_epoch=len(train_loader),
            model_type=ModelType.TEACHER,
        )
        train_teacher = False
    else:
        raise Exception(
            "Teacher training was disabled (cfg.run.teacher) and no teacher reference dir was given (cfg.reference_dir.teacher)."
        )

    ######################
    # Train teacher model #
    ######################
    if train_teacher or cfg.teacher.eval:
        # Train/retrain the teacher model on the true training set
        run_teacher(
            cfg=cfg,
            run_dir=run_dir,
            logger_wandb=logger_wandb,
            model=model_teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train_teacher,
            eval=cfg.teacher.eval,
        )

    logger.info("Finished evaluation...")

    ####################
    # Generate samples #
    ####################
    if cfg.run.sampling and not cfg.reference_dir.sampling:
        logger.info("Starting sampling...")
        samples_dir = samples_path(run_dir)
        samples_dataset = run_sampling(cfg, logger_wandb, model_teacher, samples_dir=samples_dir)
    elif cfg.reference_dir.sampling:
        logger.info(f"Sampling reference directory given: {cfg.reference_dir.sampling}")

        # Construct dataset from samples
        samples_base_dir = cfg.reference_dir.sampling
        samples_dataset = TarDataset(
            archive=os.path.join(samples_base_dir, "samples.tar"),
            eps_min=cfg.student.data.eps_min,
            eps_max=cfg.student.data.eps_max,
        )

    else:
        logger.warning(
            "Sample generation was disabled (cfg.run.sampling) and no reference dir "
            "for samples were given (cfg.reference_dir.sampling). Exiting now."
        )
        wandb.finish()

        # Do return instead of exit() since hydra decorates this method and may do something after
        # the # return
        return

    #####################
    # Train student model #
    #####################
    if not cfg.run.student:
        logger.info("Not running student model. Finishing now...")
        wandb.finish()
        return

    logger.info("Training student model on samples...")
    run_student(
        cfg,
        samples_dataset,
        logger_wandb,
        run_dir,
        train_loader,
        val_loader,
        test_loader,
    )

    # Close wandb instance. Necessary for hydra multi-runs where main() is called multipel times
    wandb.finish()


def run_student(
    cfg: DictConfig,
    samples_dataset: Dataset,
    logger_wandb: WandbLogger,
    run_dir: str,
    train_loader_data: torch.utils.data.DataLoader,
    val_loader_data: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
) -> None:
    """
    Trains a student model on a dataset of samples and evaluates it on original train/val/test loaders.

    Args:
        cfg (DictConfig): Configuration object.
        samples_dataset (Dataset): Dataset of samples to train the student model on.
        logger_wandb (WandbLogger): WandB logger.
        run_dir (str): Directory to save the checkpoint and other artifacts.
        train_loader_data (torch.utils.data.DataLoader): DataLoader for the original training set.
        val_loader_data (torch.utils.data.DataLoader): DataLoader for the original validation set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the original test set.

    Returns:
        None
    """
    # Construct dataloader from new samples
    N = len(samples_dataset)
    N_train = round(N * 0.99)
    N_val = max(1, N - N_train)
    lenghts = [N_train, N_val]
    dataset_samples_train, dataset_samples_val = random_split(samples_dataset, lengths=lenghts)
    loader_train_samples = make_dataloader_from_dataset(
        dataset=dataset_samples_train,
        num_workers=cfg.data.num_workers,
        batch_size=cfg.student.bs,
        shuffle=True,
    )
    loader_val_samples = make_dataloader_from_dataset(
        dataset=dataset_samples_val,
        num_workers=cfg.data.num_workers,
        batch_size=cfg.student.bs,
        shuffle=False,
    )

    model_student = DiscriminativeModel(
        cfg=cfg,
        learning_rate=cfg.student.lr,
        epochs=cfg.student.epochs,
        model_cfg=cfg.student,
        weight_decay=cfg.student.weight_decay,
        steps_per_epoch=len(loader_train_samples),
        model_type=ModelType.STUDENT,
    )

    accelerator, devices = get_lightning_accelerator_and_devices(cfg)

    # Setup callbacks
    callbacks = []

    # Enable rich progress bar if not in debug mode (reason: in debug mode we may want to use pdb which does not work well together when a richprogressbar is rendered)
    if not cfg.env.debug:
        callbacks.append(RichProgressBar(refresh_rate=len(loader_train_samples) // 20))

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.student.epochs,
        logger=logger_wandb,
        # accelerator="auto",
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        precision=cfg.env.precision,
        fast_dev_run=cfg.env.debug,
        profiler=cfg.env.profiler,
        default_root_dir=run_dir,
        log_every_n_steps=cfg.env.log_every_n_steps,
        enable_checkpointing=False,
    )

    # Fit model on samples and evaluate on original val loader
    val_loaders_fit = [loader_val_samples]
    if val_loader_data is not None:
        val_loaders_fit.append(val_loader_data)
    trainer.fit(model=model_student, train_dataloaders=loader_train_samples, val_dataloaders=val_loaders_fit)
    # Evaluate model
    logger.info("Evaluating student model on original train/val/test loader...")
    loaders_to_test = [train_loader_data]
    if val_loader_data is not None:
        loaders_to_test.append(val_loader_data)
    if cfg.data.eval_test:
        loaders_to_test.append(test_loader)
    trainer.test(
        model=model_student,
        dataloaders=loaders_to_test,
        verbose=True,
    )

    # Save checkpoint in general models directory to be used across experiments
    checkpoint_path = student_checkpoint_path(run_dir)
    logger.info("Saving checkpoint: " + checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)

    if "synth" in cfg.data.dataset:
        vis_student_synth_data(cfg, dataset_samples_val, logger_wandb, model_student)


from omegaconf import DictConfig
import pytorch_lightning as pl
from src.utils import teacher_checkpoint_path


def run_teacher(
    cfg: DictConfig,
    run_dir: str,
    logger_wandb: pl.loggers.WandbLogger,
    model: pl.LightningModule,
    train_loader,
    val_loader,
    test_loader,
    train: bool,
    eval: bool,
) -> None:
    """
    Trains and evaluates a teacher model.

    Args:
        cfg (DictConfig): Configuration object.
        run_dir (str): Directory to save logs and checkpoints.
        logger_wandb (pl.loggers.WandbLogger): Wandb logger object.
        model (pl.LightningModule): PyTorch Lightning module.
        train_loader (pl.LightningDataLoader): Training data loader.
        val_loader (pl.LightningDataLoader): Validation data loader.
        test_loader (pl.LightningDataLoader): Test data loader.
        train (bool): Whether to train the model.
        eval (bool): Whether to evaluate the model.
    """
    # Store number of model parameters
    summary = ModelSummary(model, max_depth=2)
    logger.info("Summary:")
    logger.info("\n" + str(summary))
    logger_wandb.experiment.config["trainable_parameters"] = summary.trainable_parameters

    accelerator, devices = get_lightning_accelerator_and_devices(cfg)

    # Setup callbacks
    callbacks = []

    # Enable rich progress bar
    callbacks.append(RichProgressBar(refresh_rate=len(train_loader) // 20))

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.teacher.epochs,
        logger=logger_wandb,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        precision=cfg.env.precision,
        fast_dev_run=cfg.env.debug,
        profiler=cfg.env.profiler,
        default_root_dir=run_dir,
        enable_checkpointing=False,
    )

    # Fit model
    if train:
        logger.info("Training new teacher model ...")
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if "synth" in cfg.data.dataset:
        data, labels = generate_data(dataset_name=cfg.data.dataset, n_samples=1000)
        plot_data_and_decision_boundary(
            pred_func=model.predict,
            data=data,
            labels=labels,
            logger_wandb=logger_wandb,
            log_key="decision-boundary/teacher-data",
            x_min=-2.0,
            x_max=2.0,
            y_min=-2.0,
            y_max=2.0,
            device=model.device,
        )

    # Evaluate model
    if eval:
        logger.info("Evaluating teacher model ...")
        loaders_to_test = [train_loader]
        if val_loader is not None:
            loaders_to_test.append(val_loader)
        if cfg.data.eval_test:
            loaders_to_test.append(test_loader)
        trainer.test(model=model, dataloaders=loaders_to_test, verbose=True)

    # Save checkpoint in general models directory to be used across experiments
    checkpoint_path = teacher_checkpoint_path(run_dir)
    logger.info("Saving checkpoint: " + checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)


def run_sampling(cfg, logger_wandb, model_teacher, samples_dir) -> Dataset:
    """
    Generates synthetic samples using a teacher model and saves them to disk.

    Args:
        cfg (omegaconf.DictConfig): Configuration object.
        logger_wandb (wandb.sdk.wandb_run.Run): WandB logger object.
        model_teacher (nn.Module): Teacher model used to generate samples.
        samples_dir (str): Directory where samples will be saved.

    Returns:
        Dataset: Dataset containing the generated samples.
    """
    # Generate samples
    device = get_devices(cfg)[0] if torch.cuda.is_available() else None
    data_shape = get_data_shape(cfg.data.dataset)

    samples_dataset = generate_samples(
        model_teacher=model_teacher,
        shape=data_shape,
        cfg=cfg,
        device=device,
        logger_wandb=logger_wandb,
        samples_dir=samples_dir,
    )

    samples_dataloader = make_dataloader_from_dataset(
        dataset=samples_dataset,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        batch_size=1000,
    )

    # Extract a batch of samples and labels
    data, labels = next(iter(samples_dataloader))
    samples = data.cpu()  # [N * 2, C, H, W]
    labels = labels.cpu()

    samples = samples.view(-1, *data_shape)  # [N*Cls, C, H, W]

    # Plot decision boundariess of synthetic 2D datasets
    if "synth" in cfg.data.dataset:
        samples = samples.view(samples.shape[0], 2)
        # Plot
        plot_data_and_decision_boundary(
            pred_func=model_teacher.predict,
            data=samples,
            labels=labels,
            logger_wandb=logger_wandb,
            log_key="decision-boundary/teacher-samples",
            device=model_teacher.device,
        )

        plot_data_and_decision_boundary(
            pred_func=model_teacher.predict,
            data=samples,
            labels=labels,
            logger_wandb=logger_wandb,
            log_key="decision-boundary/teacher-samples-small",
            device=model_teacher.device,
            x_min=samples[:, 0].min(),
            x_max=samples[:, 0].max(),
            y_min=samples[:, 1].min(),
            y_max=samples[:, 1].max(),
        )
    else:
        # Plot generated samples
        n_samples_plot = 64
        perm = torch.randperm(samples.shape[0])
        samples_subset = samples[perm[:n_samples_plot]]
        grid = torchvision.utils.make_grid(samples_subset, nrow=8, padding=2)
        logger_wandb.log_image("sampling/samples", [grid])

    return samples_dataset


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unexpected error")
        logger.exception("Finishing wandb run (wandb.finish())")
        wandb.finish()
        logger.exception("Continuing with exception")
        raise e
