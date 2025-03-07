import os
from argparse import Namespace
from typing import List
import torch
from torch.optim.adam import Adam
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.utils import data
import argparse
import logging
from tensorboardX import SummaryWriter
from peft import LoraConfig, get_peft_model  # type: ignore

from ..era5_data import utils
from ..era5_data import energy_dataset
from ..era5_data.config import cfg
from ..models.train_power import train
from ..models.test_power import test, test_baseline
from ..models.pangu_power import (
    PanguPowerPatchRecovery,
    PanguPowerConv,
)
from ..models.pangu_model import PanguModel


"""
Finetune pangu_power on the energy dataset
"""


def _setup_lora(model: torch.nn.Module, modules_to_save: List[str]) -> torch.nn.Module:
    """Sets up LoRA for the model

    Parameters
    ----------
    model : torch.nn.Module
        The model to set up LoRA for.
    modules_to_save : List[str]
        The model with LoRA setup.

    Returns
    -------
    torch.nn.Module
        Returns a Peft model, as specified in the config file.
    """

    # Get all linear layers in the model. They will be tuned by LoRA.
    target_modules = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            target_modules.append(n)

    config = LoraConfig(
        r=cfg.LORA.R,
        lora_alpha=cfg.LORA.LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=cfg.LORA.LORA_DROPOUT,
        modules_to_save=modules_to_save,
    )

    peft_model = get_peft_model(model, config)  # type: ignore
    return peft_model


def load_model(device: torch.device) -> torch.nn.Module:
    """Loads the model specified in the config file. Will also wrap model w/ LoRA if set in config.

    Parameters
    ----------
    device : torch.device
        torch device to load the model on

    Returns
    -------
    torch.nn.Module
        The loaded model
    """

    model_type = cfg.POWER.MODEL_TYPE
    req_grad_layers = []

    # Select correct model
    if model_type == "PanguPowerPatchRecovery":
        model = PanguPowerPatchRecovery(device=device).to(device)
        # Only finetune the last layer
        req_grad_layers = ["_output_power_layer"]

    elif model_type == "PanguPowerPatchRecoveryUpsample":
        model = PanguPowerPatchRecovery(device=device).to(device)
        # Finetune last two layers (output_power_layer and upsample)
        req_grad_layers = ["_output_power_layer", "upsample"]

    elif model_type == "PanguPowerConv":
        model = PanguPowerConv(device=device).to(device)
        # Only finetune the last layer
        req_grad_layers = ["_conv_power_layers"]

    elif model_type == "PanguPowerConvSigmoid":
        model = PanguPowerConv(device=device).to(device)
        # Only finetune the last layer
        req_grad_layers = ["_conv_power_layers"]

    else:
        raise ValueError(f"Model not found: {model_type}")

    # Load specified checkpoint
    if cfg.POWER.USE_CHECKPOINT:
        checkpoint = torch.load(
            cfg.POWER.CHECKPOINT, map_location=device, weights_only=False
        )

        # Setups LoRA if specified, so that the key names will match. Make sure that checkpoint is also using LoRA in that case
        if cfg.POWER.LORA:
            model = _setup_lora(model, req_grad_layers)

        model.load_state_dict(checkpoint["model"], strict=True)

    # Initialize model w/ pangu weights
    else:
        model.load_pangu_state_dict(device)

    # Set requires_grad to True for the specified layers
    for layer in req_grad_layers:
        set_requires_grad(model, layer)

        # ToDo(EliasKng): Probably wraps the model twice is checkpoint is used
        # Prepare LoRA if specified
        if cfg.POWER.LORA:
            model = _setup_lora(model, req_grad_layers)

    return model


def ddp_setup(
    rank: int, world_size: int, master_port: str, gpu_list: List[int]
) -> None:
    """Initializes the process group and sets the device for DDP

    Parameters
    ----------
        rank: int
            Unique identifier of each process
        world_size: int
            Total number of processes
        master_port: str
            Port number for the master node in the distributed setup
        gpu_list: List[int]
            List of GPUs to use
    """
    os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(gpu_list[rank])
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def create_dataloader(
    start: str,
    end: str,
    freq: str,
    batch_size: int,
    shuffle: bool,
    distributed: bool = False,
) -> data.DataLoader:
    """Creates a DataLoader for the energy dataset. If distributed is set to True, the DataLoader will be created with a DistributedSampler.

    Parameters
    ----------
    start : str
        Start date for the dataset
    end : str
        End date for the dataset
    freq : str
        Frequency of the data
    batch_size : int
        Batch size for the DataLoader
    shuffle : bool
        Whether to shuffle the data
    distributed : bool, optional
        Whether to use a DistributedSampler, by default False

    Returns
    -------
    data.DataLoader
        The DataLoader for the energy dataset
    """
    dataset = energy_dataset.EnergyDataset(
        filepath_era5=cfg.ERA5_PATH,
        filepath_power=cfg.POWER_PATH,
        startDate=start,
        endDate=end,
        freq=freq,
    )
    if not distributed:
        return data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )
    train_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler,
    )


def set_requires_grad(model: torch.nn.Module, layer_name: str) -> None:
    """Sets the `requires_grad` attribute of the parameters in the model.
    This function will first set `requires_grad` to False for all parameters in the model.
    Then, it will set `requires_grad` to True for all parameters whose names contain the specified `layer_name`.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model whose parameters' `requires_grad` attribute will be modified.
    layer_name : str
        The name (or partial name) of the layer whose parameters should have `requires_grad` set to True.
    """

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True


def setup_writer(output_path: str) -> SummaryWriter:
    """Set up a SummaryWriter for logging.

    Parameters
    ----------
    output_path : str
        The path to the directory where the writer will save logs.

    Returns
    -------
    SummaryWriter
        An instance of SummaryWriter for logging.
    """
    writer_path = os.path.join(output_path, "writer")
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    writer = SummaryWriter(writer_path)
    return writer


def setup_logger(type_net: str, horizon: int, output_path: str) -> logging.Logger:
    """Sets up the logger

    Parameters
    ----------
    type_net : str
        Used as the name of the logger.
    horizon : int
        The horizon value to be included in the logger name. Typically 24.
    output_path : str
        The path where the log file will be saved.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger_name = type_net + str(horizon)
    utils.logger_info(logger_name, os.path.join(output_path, logger_name + ".log"))
    logger = logging.getLogger(logger_name)
    return logger


def _get_device(rank: int, gpu_list: List[int]) -> torch.device:
    """Get the appropriate device (GPU or CPU) for the given rank.
    This function checks if CUDA is available and returns the corresponding
    GPU device based on the provided rank and GPU list. If CUDA is not available,
    it defaults to returning the CPU device.

    Parameters
    ----------
    rank : int
        The rank of the current process in the distributed setup.
    gpu_list : List[int]
        List of GPUs to use.

    Returns
    -------
    torch.device
        The appropriate device (GPU or CPU) for the given rank.
    """

    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_list[rank]}")
    return torch.device("cpu")


def _assert_gpu_list(gpu_list: List[int], dist: bool) -> None:
    """Asserts that the provided GPU list is valid based on the distributed training setting.

    Parameters
    ----------
    gpu_list : List[int]
        List of GPUs to use.
    dist : bool
        Whether distributed training is enabled.
    """
    assert len(gpu_list) > 0, "Please specify at least one GPU"
    if dist:
        assert (
            len(gpu_list) > 1
        ), "When distributed training is enabled, please specify at least two GPUs."
    else:
        assert (
            len(gpu_list) == 1
        ), "When distributed training is disabled, please specify exactly one GPU. If you want to use CPU, don't specify any GPU."


def main(
    rank: int, args: argparse.Namespace, world_size: int, master_port: str
) -> None:
    """Main function to set up and run the fine-tuning process for the energy dataset.

    Parameters
    ----------
    rank : int
        The rank of the current process in the distributed setup.
    args : argparse.Namespace
        Command-line arguments containing configuration parameters.
    world_size : int
        Total number of processes participating in the distributed training.
    master_port : str
        Port number for the master node in the distributed setup.

    Returns
    -------
    None
    """

    _assert_gpu_list(args.gpu_list, args.dist)

    ddp_setup(rank, world_size, master_port, args.gpu_list)

    print(f"Rank: {rank}, World Size: {world_size}")

    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)

    writer = setup_writer(output_path)
    logger = setup_logger(args.type_net, cfg.PG.HORIZON, output_path)

    device = _get_device(rank, args.gpu_list)

    if rank == 0:
        logger.info(f"Start finetuning {args.type_net} on energy dataset")

    train_dataloader = create_dataloader(
        cfg.PG.TRAIN.START_TIME,
        cfg.PG.TRAIN.END_TIME,
        cfg.PG.TRAIN.FREQUENCY,
        cfg.PG.TRAIN.BATCH_SIZE,
        True,
        args.dist,
    )
    val_dataloader = create_dataloader(
        cfg.PG.VAL.START_TIME,
        cfg.PG.VAL.END_TIME,
        cfg.PG.VAL.FREQUENCY,
        cfg.PG.VAL.BATCH_SIZE,
        False,
    )

    model = load_model(device)
    model = DDP(model, device_ids=[device])

    # If static graph is not set, LoRA returns errors.
    if cfg.POWER.LORA:
        model._set_static_graph()

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.PG.TRAIN.LR,
        weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY,
    )
    if rank == 0:
        msg = "\n"
        msg += utils.torch_summarize(model, show_weights=False)
        logger.info(msg)

    # Print test name again after model summary (it is very long)
    if rank == 0:
        logger.info(f"Start finetuning {args.type_net} on energy dataset")

    torch.set_num_threads(cfg.GLOBAL.NUM_STREADS)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50], gamma=0.5
    )
    start_epoch = args.start_epoch

    # Manually step the scheduler to the correct epoch
    if start_epoch > 1:
        for epoch in range(start_epoch - 1):
            print(f"Step: {epoch}")
            lr_scheduler.step(epoch)

    model = train(
        model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        res_path=output_path,
        writer=writer,
        logger=logger,
        start_epoch=start_epoch,
        rank=rank,
        device=device,
    )

    destroy_process_group()


def test_best_model(args: argparse.Namespace) -> None:
    """Tests the best model (model that has the lowest validation loss) on the test dataset.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments containing the configuration for testing the model. Is called after training.

    Returns
    -------
    None
    """
    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)
    logger = setup_logger(args.type_net.split("/")[-1], cfg.PG.HORIZON, output_path)
    logger.info(f"Begin testing: {args.type_net}")
    device = _get_device(0, args.gpu_list)

    best_model = torch.load(
        os.path.join(output_path, "models/best_model.pth"),
        map_location=device,
        weights_only=False,
    ).to(device)

    set_model_device_recursively(best_model, device)

    test_dataloader = create_dataloader(
        cfg.PG.TEST.START_TIME,
        cfg.PG.TEST.END_TIME,
        cfg.PG.TEST.FREQUENCY,
        cfg.PG.TEST.BATCH_SIZE,
        False,
    )

    test(
        test_loader=test_dataloader,
        model=best_model,
        device=device,
        res_path=output_path,
        logger=logger,
    )


def set_model_device_recursively(module: nn.Module, device: torch.device) -> None:
    """Recursively sets the `device` attribute for the given module and all its children. This is required becuase some masks are generated dynamically during model inference using the self.device parameter of that layers, which is set initially during model instantiation. If e.g., training and testing happens on different, the masks will be generated on the wrong device (if not set correctly by this function) which will cause an error.

    Args:
        module (nn.Module): The root module whose `device` attribute and its children's will be updated.
        device (torch.device): The target device to set.
    """
    if hasattr(module, "device"):
        module.device = device  # type: ignore

    for child in module.children():
        set_model_device_recursively(child, device)


def test_baselines(args: Namespace, baseline_type: str) -> None:
    """Test the performance of baseline models.

    Parameters
    ----------
    args : Namespace
        Contains passed arguments when starting the script.
    baseline_type : str
        Specifies the type of baseline prediction, can be "formula", "persistence" or "mean".

    Returns
    -------
    None
    """
    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)
    logger = setup_logger(args.type_net, cfg.PG.HORIZON, output_path)
    logger.info("Begin testing...")
    device = _get_device(0, args.gpu_list)

    test_dataloader = create_dataloader(
        cfg.PG.TEST.START_TIME,
        cfg.PG.TEST.END_TIME,
        cfg.PG.TEST.FREQUENCY,
        cfg.PG.TEST.BATCH_SIZE,
        False,
    )

    pangu_model = PanguModel(device=device).to(device)

    checkpoint = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, weights_only=False)
    pangu_model.load_state_dict(checkpoint["model"])

    test_baseline(
        test_loader=test_dataloader,
        pangu_model=pangu_model,
        device=device,
        res_path=output_path,
        baseline_type=baseline_type,
    )
