import os
from datetime import datetime
import warnings
from wind_fusion.pangu_pytorch.era5_data import utils, utils_data, score
from wind_fusion.pangu_pytorch.models.train_power import (
    model_inference_power,
    model_inference_pangu,
    baseline_inference,
    load_land_sea_mask,
    visualize,
)
from wind_fusion.pangu_pytorch.models.baseline_formula import BaselineFormula


warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
)

warnings.filterwarnings(
    "ignore",
    message="Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas",
)


def calculate_scores(
    output_power, target_power, lsm_expanded, mean_power_per_grid_point, target_time
):
    """
    Calculates RMSE, MAE, and ACC scores for power predictions.
    """
    scores = {}

    # Mask outputs and targets
    output_power_masked = output_power[lsm_expanded.squeeze() == 1]
    target_power_masked = target_power[lsm_expanded.squeeze() == 1]

    # RMSE
    scores["rmse"] = (
        (score.rmse(output_power_masked, target_power_masked)).detach().cpu().numpy()
    )

    # Mean absolute error (MAE)
    scores["mae"] = (
        (score.mae(output_power_masked, target_power_masked)).detach().cpu().numpy()
    )

    # Calculate power anomalies
    output_power_anomaly = output_power - mean_power_per_grid_point
    target_power_anomaly = target_power - mean_power_per_grid_point

    # Mask anomalies
    output_power_anomaly_masked = output_power_anomaly.squeeze(0)[
        lsm_expanded.squeeze() == 1
    ]
    target_power_anomaly_masked = target_power_anomaly.squeeze(0)[
        lsm_expanded.squeeze() == 1
    ]

    # ACC
    scores["acc"] = (
        (
            score.weighted_acc(
                output_power_anomaly_masked.detach().cpu(),
                target_power_anomaly_masked.detach().cpu(),
                weighted=False,
            )
        )
        .detach()
        .cpu()
        .numpy()
    )

    return target_time, scores


def test(test_loader, model, device, res_path):
    rmse_power = dict()
    mae_power = dict()
    acc_power = dict()

    aux_constants = utils_data.loadAllConstants(device=device)

    for id, data in enumerate(test_loader, 0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] predict on {id}")
        (
            input_upper_test,
            input_surface_test,
            input_power_test,
            target_power_test,
            target_upper_test,
            target_surface_test,
            periods_test,
        ) = data

        input_upper_test, input_surface_test, target_power_test = (
            input_upper_test.to(device),
            input_surface_test.to(device),
            target_power_test.to(device),
        )
        model.eval()

        # Inference
        output_power_test = model_inference_power(
            model, input_upper_test, input_surface_test, aux_constants
        )

        # Apply lsm
        lsm_expanded = load_land_sea_mask(output_power_test.device, fill_value=0)
        output_power_test = output_power_test * lsm_expanded

        # Visualize
        target_time = periods_test[1][0]
        png_path = os.path.join(res_path, "png")
        utils.mkdirs(png_path)
        visualize(
            output_power_test,
            target_power_test,
            input_surface_test,
            input_upper_test,
            target_surface_test,
            target_upper_test,
            target_time,
            png_path,
        )

        # Compute test scores
        output_power_test = output_power_test.squeeze()
        target_power_test = target_power_test.squeeze()
        mean_power_per_grid_point = utils_data.loadMeanPower(output_power_test.device)

        # Calculate scores using the helper function
        target_time, scores = calculate_scores(
            output_power_test,
            target_power_test,
            lsm_expanded,
            mean_power_per_grid_point,
            target_time,
        )

        # Update score dictionaries
        rmse_power[target_time] = scores["rmse"]
        mae_power[target_time] = scores["mae"]
        acc_power[target_time] = scores["acc"]

    # Save scores to csv
    csv_path = os.path.join(res_path, "csv")
    utils.mkdirs(csv_path)
    utils.save_error_power(csv_path, rmse_power, "rmse")
    utils.save_error_power(csv_path, mae_power, "mae")
    utils.save_error_power(csv_path, acc_power, "acc")


def test_baseline(test_loader, pangu_model, device, res_path, baseline_type: str):
    rmse_power = dict()
    mae_power = dict()
    acc_power = dict()

    baseline_formula = BaselineFormula(device).to(device)

    for id, data in enumerate(test_loader, 0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] predict on {id}")
        (
            input_test,
            input_surface_test,
            input_power_test,
            target_power_test,
            target_upper_test,
            target_surface_test,
            periods_test,
        ) = data

        input_test, input_power_test, input_surface_test, target_power_test = (
            input_test.to(device),
            input_power_test.to(device),
            input_surface_test.to(device),
            target_power_test.to(device),
        )

        # Inference
        mean_power = utils_data.loadMeanPower(device)

        # Pangu output is required for formula baseline, therefore we need to run the model
        if baseline_type == "formula":
            pangu_model.eval()
            # Inference
            aux_constants = utils_data.loadAllConstants(device=device)
            output_weather_upper, output_weather_surface = model_inference_pangu(
                pangu_model, input_test, input_surface_test, aux_constants
            )

            # Inference
            output_power_test = baseline_inference(
                input_power_test,
                mean_power,
                output_weather_upper,
                output_weather_surface,
                baseline_formula,
                baseline_type,
            )
        else:
            output_power_test = baseline_inference(
                input_power_test, mean_power, type=baseline_type
            )

        # Apply lsm
        lsm_expanded = load_land_sea_mask(output_power_test.device, fill_value=0)
        output_power_test = output_power_test * lsm_expanded

        # Visualize
        target_time = periods_test[1][0]
        png_path = os.path.join(res_path, "png")

        # This can be used to pre-generate pangu outputs, which are required for some visualizations
        # save_output_pth(output_weather_upper, output_weather_surface, target_time, res_path)

        utils.mkdirs(png_path)
        visualize(
            output_power_test,
            target_power_test,
            input_surface_test,
            input_test,
            target_surface_test,
            target_upper_test,
            target_time,
            png_path,
            input_power=input_power_test,
        )

        # Compute test scores
        output_power_test = output_power_test.squeeze()
        target_power_test = target_power_test.squeeze()
        mean_power_per_grid_point = utils_data.loadMeanPower(output_power_test.device)

        # Calculate scores using the helper function
        target_time, scores = calculate_scores(
            output_power_test,
            target_power_test,
            lsm_expanded,
            mean_power_per_grid_point,
            target_time,
        )

        # Update score dictionaries
        rmse_power[target_time] = scores["rmse"]
        mae_power[target_time] = scores["mae"]
        acc_power[target_time] = scores["acc"]

    # Save scores to csv
    csv_path = os.path.join(res_path, "csv")
    utils.mkdirs(csv_path)
    utils.save_error_power(csv_path, rmse_power, "rmse")
    utils.save_error_power(csv_path, mae_power, "mae")
    utils.save_error_power(csv_path, acc_power, "acc")
