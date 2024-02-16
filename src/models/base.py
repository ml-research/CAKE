import os
import sys

from src.models.cnn import CnnWithAdaptivePool, Cnn, CnnBig, CnnTiny
from src.models.mlp import MLP
from src.models.svm import PrimalLinearSVM, PrimalPolySVM, PrimalRBFSVM

import timm
from omegaconf import DictConfig
from torch import nn

from src.models.lenet import LeNet5, LeNet5Half
from src.data import get_number_of_classes, get_data_shape
from src.models.resnet import resnet
from src.models.vit import ViT
from src.models.logistic_regression import LogisticRegression


def make_model(cfg: DictConfig, model_cfg: DictConfig):
    """
    Get the model based on the architecture string

    Args:
        cfg: General config.
        model_cfg: Specific model config.

    Returns:
        Instantiated model object
    """
    num_classes = get_number_of_classes(cfg.data.dataset)
    in_channels = model_cfg.in_channels
    try:
        if model_cfg.arch == "resnet":
            arch = "resnet" + str(model_cfg.resnet.depth)
        else:
            arch = model_cfg.arch

        # First try if model arch is from timm
        model = timm.create_model(
            arch,
            pretrained=model_cfg.pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )
        # NOTE: Not sure where I got the following overrides from (conv1, maxpool)??? Without this,
        # the model performs significantly worse (random init for the input layer is probably better?)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        return model
    except RuntimeError as e:
        data_shape = get_data_shape(cfg.data.dataset)
        in_features_linear = data_shape[0] * data_shape[1] * data_shape[2]
        if model_cfg.arch == "mlp":
            return MLP(
                in_features=in_features_linear,
                num_classes=get_number_of_classes(cfg.data.dataset),
                hidden_size=model_cfg.mlp.hidden_size,
                num_hidden=model_cfg.mlp.num_hidden,
            )
        elif model_cfg.arch == "lenet":
            return LeNet5(
                in_channels=in_channels,
                num_classes=num_classes,
            )
        elif model_cfg.arch == "lenet-half":
            return LeNet5Half(
                in_channels=in_channels,
                num_classes=num_classes,
            )

        elif model_cfg.arch == "cnn":
            return Cnn(
                in_channels=in_channels,
                num_classes=num_classes,
            )
        elif model_cfg.arch == "cnn-tiny":
            return CnnTiny(
                in_channels=in_channels,
                num_classes=num_classes,
            )
        elif model_cfg.arch == "cnn-big":
            return CnnBig(
                in_channels=in_channels,
                num_classes=num_classes,
            )
        elif model_cfg.arch == "cnn_adaptive_pool":
            return CnnWithAdaptivePool(
                in_channels=in_channels,
                last_layer_dim=model_cfg.cnn_adaptive_pool.last_layer_dim,
                num_classes=num_classes,
            )
        elif model_cfg.arch == "resnet" and model_cfg.resnet.depth == 4:
            return resnet(
                depth=model_cfg.resnet.depth,
                num_classes=num_classes,
                in_channels=in_channels,
                last_layer_dim=64,  # tiny (mnist)
                # last_layer_dim=256,  # cifar10
            )
        elif model_cfg.arch == "vit":
            return ViT(
                image_size=data_shape[1],
                channels=data_shape[0],
                patch_size=model_cfg.vit.patch_size,
                num_classes=num_classes,
                dim=model_cfg.vit.dim,
                depth=model_cfg.vit.depth,
                heads=model_cfg.vit.heads,
                mlp_dim=model_cfg.vit.mlp_dim,
                dropout=model_cfg.vit.dropout,
                emb_dropout=model_cfg.vit.emb_dropout,
            )
        elif model_cfg.arch == "logistic_regression":
            return LogisticRegression(
                input_dim=in_features_linear,
                output_dim=num_classes,
            )
        elif model_cfg.arch == "svm":
            if model_cfg.svm.kernel == "linear":
                return PrimalLinearSVM(num_features=in_features_linear, C=model_cfg.svm.C, num_classes=num_classes)
            elif model_cfg.svm.kernel == "rbf":
                return PrimalRBFSVM(
                    num_features=in_features_linear,
                    C=model_cfg.svm.C,
                    gamma=model_cfg.svm.rbf.gamma,
                    num_fourier_features=model_cfg.svm.rbf.num_fourier_features,
                    num_classes=num_classes,
                )
            elif model_cfg.svm.kernel == "poly":
                return PrimalPolySVM(
                    num_features=in_features_linear,
                    C=model_cfg.svm.C,
                    degree=model_cfg.svm.poly.degree,
                    num_classes=num_classes,
                )
            else:
                raise ValueError(f"Unknown kernel: {model_cfg.svm.kernel} -- has to be one of 'linear', 'rbf', 'poly'.")

        else:
            raise ValueError(f"Unknown model arch: {model_cfg.arch}")


if __name__ == "__main__":
    import sys

    models = {}
    for d in [18, 34, 50, 101, 152]:
        model = timm.create_model(
            "resnet" + str(d),
            pretrained=False,
            num_classes=10,
            in_chans=3,
        )
        models["resnet-" + str(d)] = model

    models["resnet-4"] = resnet(
        depth=4,
        num_classes=10,
        in_channels=3,
        last_layer_dim=256,  # tiny (mnist)
    )
    models["vit-8"] = ViT(
        image_size=32,
        channels=3,
        patch_size=4,
        num_classes=10,
        dim=64,
        depth=8,
        heads=8,
        mlp_dim=64,
        dropout=0.0,
        emb_dropout=0.0,
    )
    models["vit-4"] = ViT(
        image_size=32,
        channels=3,
        patch_size=4,
        num_classes=10,
        dim=64,
        depth=4,
        heads=8,
        mlp_dim=64,
        dropout=0.0,
        emb_dropout=0.0,
    )
    models["vit-3"] = ViT(
        image_size=32,
        channels=1,
        patch_size=4,
        num_classes=10,
        dim=32,
        depth=4,
        heads=4,
        mlp_dim=32,
        dropout=0.0,
        emb_dropout=0.0,
    )
    models["cnn"] = Cnn(
        in_channels=1,
        num_classes=10,
    )
    models["mlp"] = MLP(in_features=768, num_classes=10, hidden_size=100, num_hidden=4)

    models["lenet-5"] = LeNet5(in_channels=1, num_classes=10)
    models["lenet-5-half"] = LeNet5Half(in_channels=1, num_classes=10)
    models["vgg11"] = timm.create_model(
        "vgg11",
        pretrained=False,
        num_classes=10,
        in_chans=3,
    )

    for arch, model in models.items():
        print(arch, sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3)
