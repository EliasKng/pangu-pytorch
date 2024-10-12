import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("/hkfs/home/project/hk-project-test-mlperf/om1434/masterarbeit")
from wind_fusion import energy_dataset

from era5_data import utils
from era5_data.config import cfg
import torch
from torch.optim.adam import Adam
import os
from torch.utils import data
from models.pangu_power_sample import test, train
from models.pangu_power import PanguPowerPatchRecovery, PanguPowerConv
import argparse
import time
import logging
from tensorboardX import SummaryWriter

"""
Finetune pangu_power on the energy dataset
"""


def setup_model(type: str):
    """Loads the specified model and sets requires_grad

    Parameters
    ----------
    type : str
        Which model to load
    """
    if type == "PanguPowerPatchRecovery":
        model = PanguPowerPatchRecovery(device=device).to(device)

        checkpoint = torch.load(
            cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device, weights_only=False
        )

        pretrained_dict = checkpoint["model"]
        model_dict = model.state_dict()

        # Filter out keys in pretrained_dict that belong to _output_layer (conv and conv_surface)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if "_output_layer" not in k
        }

        # Update the model's state_dict except the _output_layer
        model_dict.update(pretrained_dict)

        # Load the updated state_dict into the model
        model.load_state_dict(model_dict)

        # Only finetune the last layer
        set_requires_grad(model, "_output_layer")

    elif type == "PanguPowerConv":
        model = PanguPowerConv(device=device).to(device)
        checkpoint = torch.load(
            "/home/hk-project-test-mlperf/om1434/masterarbeit/wind_fusion/pangu_pytorch/result/PanguPowerConv_64_128_64_1_k3/24/models/train_4.pth",
            map_location=device,
            weights_only=False,
        )
        print("Loaded pangu power conv model")
        model.load_state_dict(checkpoint["model"])

        # Only finetune the last layer
        set_requires_grad(model, "_conv_power_layers")

    else:
        raise ValueError("Model not found")

    return model


def create_dataloader(start, end, freq, batch_size, shuffle):
    dataset = energy_dataset.EnergyDataset(
        filepath_era5=cfg.ERA5_PATH,
        filepath_power=cfg.POWER_PATH,
        startDate=start,
        endDate=end,
        freq=freq,
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def set_requires_grad(model, layer_name):
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True
            print("Requires grad: ", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type_net", type=str, default="PanguPowerConv_64_128_64_1_k3_2"
    )
    parser.add_argument("--load_my_best", type=bool, default=True)
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--dist", default=False)

    args = parser.parse_args()
    starts = time.time()

    PATH = cfg.PG_INPUT_PATH

    opt = {
        "gpu_ids": list(range(torch.cuda.device_count()))
    }  # Automatically select available GPUs
    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    print(f"Available GPUs: {gpu_list}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Predicting on {device}")

    output_path = os.path.join(cfg.PG_OUT_PATH, args.type_net, str(cfg.PG.HORIZON))
    utils.mkdirs(output_path)

    writer_path = os.path.join(output_path, "writer")
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)

    writer = SummaryWriter(writer_path)

    logger_name = args.type_net + str(cfg.PG.HORIZON)
    utils.logger_info(logger_name, os.path.join(output_path, logger_name + ".log"))

    logger = logging.getLogger(logger_name)

    train_dataloader = create_dataloader(
        cfg.PG.TRAIN.START_TIME,
        cfg.PG.TRAIN.END_TIME,
        cfg.PG.TRAIN.FREQUENCY,
        cfg.PG.TRAIN.BATCH_SIZE,
        True,
    )
    val_dataloader = create_dataloader(
        cfg.VAL.START_TIME,
        cfg.VAL.END_TIME,
        cfg.PG.VAL.FREQUENCY,
        cfg.PG.VAL.BATCH_SIZE,
        False,
    )
    test_dataloader = create_dataloader(
        cfg.TEST.START_TIME,
        cfg.TEST.END_TIME,
        cfg.PG.TEST.FREQUENCY,
        cfg.PG.TEST.BATCH_SIZE,
        False,
    )

    model = setup_model("PanguPowerConv")

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.PG.TRAIN.LR,
        weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY,
    )

    msg = "\n"
    msg += utils.torch_summarize(model, show_weights=False)
    logger.info(msg)

    torch.set_num_threads(cfg.GLOBAL.NUM_STREADS)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50], gamma=0.5
    )
    start_epoch = 1

    model = train(
        model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        res_path=output_path,
        device=device,
        writer=writer,
        logger=logger,
        start_epoch=start_epoch,
    )

    if args.load_my_best:
        best_model = torch.load(
            os.path.join(output_path, "models/best_model.pth"), map_location="cuda:0"
        )

    logger.info("Begin testing...")

    test(
        test_loader=test_dataloader,
        model=best_model,
        device=device,
        res_path=output_path,
    )
    # CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 finetune_lastLayer_ddp.py --dist True
