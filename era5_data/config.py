from types import SimpleNamespace as ConfigNamespace
import os
import torch

__C = ConfigNamespace()
cfg = __C
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

__C.GLOBAL = ConfigNamespace()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.BATCH_SZIE = 1

for dirs in [__C.ROOT_DIR, "your_path"]:
    if os.path.exists(dirs):
        __C.GLOBAL.PATH = dirs
assert __C.GLOBAL.PATH is not None
__C.GLOBAL.SEED = 99
__C.GLOBAL.NUM_STREADS = 16

# Paths
__C.PG_INPUT_PATH = os.path.join(__C.ROOT_DIR, "data")
assert __C.PG_INPUT_PATH is not None

__C.PG_OUT_PATH = os.path.join(__C.GLOBAL.PATH, "result")
assert __C.PG_OUT_PATH is not None

__C.ERA5_PATH = "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
__C.POWER_PATH = (
    "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/om1434/offshore/offshore.zarr"
)
# Land sea mask path
__C.LSM_PATH = (
    "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/om1434/offshore/lsm_global.zarr"
)
# Mean Power path
__C.MEAN_POWER_PATH = "/home/hk-project-test-mlperf/om1434/masterarbeit/wind_fusion/pangu_pytorch/data/aux_data/mean_power_per_grid_point.npy"
__C.POWER_CURVE_PATH = "/home/hk-project-test-mlperf/om1434/masterarbeit/wind_fusion/pangu_pytorch/data/power_curves/wind_turbine_power_curves.csv"

# Pangu pre-inferenced outputs: outputs that have been pre-inferenced with Pangu and are used for visualization
__C.PANGU_INFERENCE_OUTPUTS = (
    "/lsdf/kit/imk-tro/projects/Gruppe_Quinting/om1434/pangu_outputs"
)

__C.ERA5_UPPER_LEVELS = [
    "1000",
    "925",
    "850",
    "700",
    "600",
    "500",
    "400",
    "300",
    "250",
    "200",
    "150",
    "100",
    "50",
]
__C.ERA5_SURFACE_VARIABLES = ["msl", "u10", "v10", "t2m"]
__C.ERA5_UPPER_VARIABLES = ["z", "q", "t", "u", "v"]

__C.PG = ConfigNamespace()
__C.PG.HORIZON = 24  # Forecast horizon
# Use land sea mask when calculating loss (set for all: train, val, test)
__C.PG.USE_LSM = True

__C.PG.TRAIN = ConfigNamespace()
__C.PG.TRAIN.EPOCHS = 100
__C.PG.TRAIN.LR = 1e-4  # 5e-6  # 5e-4
__C.PG.TRAIN.WEIGHT_DECAY = 1e-4  # 3e-6
__C.PG.TRAIN.START_TIME = "20160101"
__C.PG.TRAIN.END_TIME = "20161231"
# __C.PG.TRAIN.END_TIME = "20160102"
__C.PG.TRAIN.FREQUENCY = "6h"
__C.PG.TRAIN.BATCH_SIZE = 1  # Per used GPU
__C.PG.TRAIN.UPPER_WEIGHTS = [3.00, 0.60, 1.50, 0.77, 0.54]
__C.PG.TRAIN.SURFACE_WEIGHTS = [1.50, 0.77, 0.66, 3.00]
__C.PG.TRAIN.SAVE_INTERVAL = 5
__C.PG.TRAIN.USE_LSM = __C.PG.USE_LSM

__C.PG.VAL = ConfigNamespace()
__C.PG.VAL.START_TIME = "20170101"
__C.PG.VAL.END_TIME = "20171231"
# __C.PG.VAL.END_TIME = "20170108"
__C.PG.VAL.FREQUENCY = "48h"
__C.PG.VAL.BATCH_SIZE = 1
__C.PG.VAL.INTERVAL = 1
__C.PG.VAL.USE_LSM = __C.PG.USE_LSM

__C.PG.TEST = ConfigNamespace()
__C.PG.TEST.START_TIME = "20180101"
__C.PG.TEST.END_TIME = "20181231"
# __C.PG.TEST.END_TIME = "20180108"
__C.PG.TEST.FREQUENCY = "48h"
__C.PG.TEST.BATCH_SIZE = 1
__C.PG.TEST.USE_LSM = __C.PG.USE_LSM

__C.PG.BENCHMARK = ConfigNamespace()

__C.PG.BENCHMARK.PRETRAIN_24 = os.path.join(
    __C.PG_INPUT_PATH, "pretrained_model/pangu_weather_24.onnx"
)
__C.PG.BENCHMARK.PRETRAIN_6 = os.path.join(
    __C.PG_INPUT_PATH, "pretrained_model/pangu_weather_6.onnx"
)
__C.PG.BENCHMARK.PRETRAIN_3 = os.path.join(
    __C.PG_INPUT_PATH, "pretrained_model/pangu_weather_3.onnx"
)
__C.PG.BENCHMARK.PRETRAIN_1 = os.path.join(
    __C.PG_INPUT_PATH, "pretrained_model/pangu_weather_1.onnx"
)

__C.PG.BENCHMARK.PRETRAIN_24_fp16 = os.path.join(
    __C.PG_INPUT_PATH, "pretrained_model_fp16/pangu_weather_24_fp16.onnx"
)

__C.PG.BENCHMARK.PRETRAIN_24_torch = os.path.join(
    __C.PG_INPUT_PATH, "pretrained_model/pangu_weather_24_torch.pth"
)

__C.POWER = ConfigNamespace()

# Specifies if the model should be trained from scratch (pretrained pangu weights will be used) or if a checkpoint should be used.
__C.POWER.USE_CHECKPOINT = False
# If POWER.USE_CHECKPOINT == True: Select the checkpoint to start the training from. The model is loaded from the checkpoint.
__C.POWER.CHECKPOINT = ""
# Specify the type of model to be initialized, must match the model type in the checkpoint
# Can be:
# - PanguPowerPatchRecovery: Replaces the patch recovery layer of pangu with a new convolution that aims to predict power
# - PanguPowerConv: Adds convolutional layers to the output of pangu to use pangus output to predict power
# - PanguPowerConvSigmoid: Same as PanguPowerConv but with a sigmoid activation function at the end
__C.POWER.MODEL_TYPE = "PanguPowerConv"


# ***** LORA *****
# Contains hyperparameters for LORA. Works best with MODEL_TYPE="PanguPowerPatchRecovery".
__C.POWER.LORA = False  # Whether to use LORA. If POWER.USE_CHECKPOINT == True, the checkpoint must have been trained with LORA, too.

__C.LORA = ConfigNamespace()
__C.LORA.R = 4
__C.LORA.LORA_ALPHA = 8
__C.LORA.LORA_DROPOUT = 0.3


# ***** PowerConv *****
# Contains hyperparameters for PanguPowerConv
__C.POWERCONV = ConfigNamespace()
__C.POWERCONV.IN_CHANNELS = 28
__C.POWERCONV.OUT_CHANNELS = [64, 128, 64, 1]
__C.POWERCONV.KERNEL_SIZE = 3
__C.POWERCONV.STRIDE = 1
__C.POWERCONV.PADDING = 1
# First convolutional layer may have different kernel size and padding
__C.POWERCONV.KERNEL_SIZE_FIRST = 1
__C.POWERCONV.PADDING_FIRST = 0


# Contains the power curve of Vestas Offshore V164-8000, which is used to calculate power from wind speed in the CDS dataset:
# Power curves can be found at:
# https://confluence.ecmwf.int/display/CKB/Climate+and+energy+indicators+for+Europe+datasets%3A+Technical+description+of+methodologies+followed+in+the+development+of+each+product
__C.POWER_CURVE_OFFSHORE = {
    0: 0.0,
    3.5: 0.0,
    4: 0.00875,
    4.5: 0.01875,
    5: 0.035,
    5.5: 0.063125,
    6: 0.09375,
    6.5: 0.1375,
    7: 0.18125,
    7.5: 0.240625,
    8: 0.3,
    8.5: 0.38625,
    9: 0.4725,
    9.5: 0.58625,
    10: 0.7,
    10.5: 0.79875,
    11: 0.8975,
    11.5: 0.95,
    12: 0.96875,
    12.5: 0.990625,
    13: 1.0,
    25: 1.0,
    25.000000001: 0.0,
    500: 0.0,
}
# Same, but here in kW instead of capacity factors
__C.POWER_CURVE_OFFSHORE_kW = {
    # No power below 3.5 m/s
    0: 0,
    3.5: 0,
    4: 70,
    4.5: 150,
    5: 280,
    5.5: 505,
    6: 750,
    6.5: 1100,
    7: 1450,
    7.5: 1925,
    8: 2400,
    8.5: 3090,
    9: 3780,
    9.5: 4690,
    10: 5600,
    10.5: 6390,
    11: 7180,
    11.5: 7600,
    12: 7750,
    12.5: 7925,
    # Max power at 12.5-25 m/s
    13: 8000,
    25: 8000,
    # Power cut-off at 25 m/s
    25.000000001: 0,
    500: 0.0,
}
