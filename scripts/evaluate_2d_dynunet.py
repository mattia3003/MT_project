import os
import glob
import argparse
from pathlib import Path
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    Resized,
    NormalizeIntensityd,
    EnsureTyped,
    SaveImaged
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import DynUNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric


def pair_files(data_root):
    pairs = {}
    for image_path in glob.glob(os.path.join(data_root, "**", "bravo", "*.nii*"), recursive=True):
        relative = os.path.relpath(image_path, data_root).split(os.sep)
        stem = os.path.basename(image_path).replace(".nii.gz", "").replace(".nii", "")
        stem = stem.removesuffix("_bravo").removesuffix(".bravo")
        group = relative[-3] if len(relative) >= 3 else stem
        pairs[(group, stem)] = image_path

    labels = {}
    for label_path in glob.glob(os.path.join(data_root, "**", "seg", "*.nii*"), recursive=True):
        relative = os.path.relpath(label_path, data_root).split(os.sep)
        stem = os.path.basename(label_path).replace(".nii.gz", "").replace(".nii", "")
        stem = stem.removesuffix("_seg").removesuffix(".seg")
        group = relative[-3] if len(relative) >= 3 else stem
        labels[(group, stem)] = label_path

    if set(pairs) != set(labels):
        raise ValueError("BRAVO/SEG files could not be paired exactly in the evaluation directory.")
    return [{"image": pairs[key], "label": labels[key]} for key in sorted(pairs)]

def main():
    # --------------------------------------------------------------------------
    # 0. ARGUMENT PARSER (THRESHOLD & SAVING CONTROLS)
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Evaluate 2D DynUNet Model on Test Set")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5, 
        help="Probability threshold for binarizing predictions (default: 0.5)"
    )
    parser.add_argument(
        "--save_preds", 
        action="store_true", 
        help="Enable saving of predicted .nii.gz files to disk (disabled by default)"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="best_synthetic_model_TS256_40epochs_dynunet_10_0.pth", 
        help="Path to model weights file"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing paired bravo/ and seg/ folders")
    parser.add_argument("--target_size", type=int, nargs=2, default=(256, 256), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--resize_mode", choices=("pad_crop", "resize"), default="pad_crop")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # 1. PATH DEFINITION & TEST DATASET ISOLATION
    # --------------------------------------------------------------------------
    test_data_root = args.data_dir
    output_predictions_dir = "../models/predictions"
    model_weights_path = args.model_path

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights file '{model_weights_path}' not found. Train the model first!")

    test_files = pair_files(test_data_root)
    patient_count = len({Path(item["image"]).parent.parent.name for item in test_files})

    print(f"Test Dataset Loaded: {patient_count} groups | {len(test_files)} total 2D slices")
    print(f"Prediction Threshold: {args.threshold}")
    print(f"Save Predictions to Disk: {args.save_preds}")

    # --------------------------------------------------------------------------
    # 2. EVALUATION TRANSFORM PIPELINE
    # --------------------------------------------------------------------------
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=tuple(args.target_size), mode=("constant", "constant"))
        if args.resize_mode == "pad_crop"
        else Resized(keys=["image", "label"], spatial_size=tuple(args.target_size), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"])
    ])

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)

    # --------------------------------------------------------------------------
    # 3. RE-CREATE MODEL & LOAD WEIGHTS
    # --------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    model = DynUNet(
            spatial_dims=2, in_channels=1, out_channels=1,
            kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]], 
            strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]],
            upsample_kernel_size=[[2, 2], [2, 2], [2, 2], [2, 2]],
            filters=(16, 32, 64, 128, 256),
            dropout=0.0,
            norm_name="instance",
            deep_supervision=True
        ).to(device)

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    dice_metrics = {
        "with_background": DiceMetric(include_background=True, reduction="mean"),
        "lesion_only": DiceMetric(include_background=False, reduction="mean"),
    }
    hd95_metrics = {
        "with_background": HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=95, reduction="mean"),
        "lesion_only": HausdorffDistanceMetric(include_background=False, distance_metric="euclidean", percentile=95, reduction="mean"),
    }
    empty_prediction_counts = {0: 0, 1: 0}
    empty_label_counts = {0: 0, 1: 0}
    empty_prediction_slices = set()
    empty_label_slices = set()

    # --------------------------------------------------------------------------
    # 4. INFERENCE & METRIC EVALUATION LOOP
    # --------------------------------------------------------------------------
    if args.save_preds:
        saver = SaveImaged(
            keys="pred",
            meta_keys="pred_meta_dict",
            output_dir=output_predictions_dir,
            output_postfix="pred",
            output_ext=".nii.gz",
            resample=False,
            separate_folder=False
        )

    print("\nRunning inference on test dataset...")
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            test_inputs = batch_data["image"].to(device)
            test_labels = batch_data["label"].to(device)

            outputs = model(test_inputs)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            elif isinstance(outputs, torch.Tensor) and outputs.ndim == 5:
                outputs = outputs[:, 0]  # Take primary high-res prediction head

            # Apply Sigmoid and binarize using the custom threshold
            preds = (torch.sigmoid(outputs) > args.threshold).float()
            metric_preds = torch.cat([1.0 - preds, preds], dim=1)
            metric_labels = torch.cat([1.0 - test_labels, test_labels], dim=1)

            for class_index in (0, 1):
                prediction_is_empty = not metric_preds[:, class_index].any().item()
                label_is_empty = not metric_labels[:, class_index].any().item()
                if prediction_is_empty:
                    empty_prediction_counts[class_index] += 1
                    empty_prediction_slices.add(i)
                if label_is_empty:
                    empty_label_counts[class_index] += 1
                    empty_label_slices.add(i)

            for name in dice_metrics:
                dice_metrics[name](y_pred=metric_preds, y=metric_labels)
                try:
                    hd95_metrics[name](y_pred=metric_preds, y=metric_labels)
                except Exception:
                    pass

            # Save predictions ONLY if --save_preds flag is set
            if args.save_preds:
                batch_data["pred"] = preds
                batch_data["pred_meta_dict"] = batch_data["image_meta_dict"]
                batch_list = decollate_batch(batch_data)
                for item in batch_list:
                    saver(item)

    # --------------------------------------------------------------------------
    # 5. FINAL REPORTING
    # --------------------------------------------------------------------------
    final_dice = {name: metric.aggregate().item() for name, metric in dice_metrics.items()}
    for metric in dice_metrics.values():
        metric.reset()
    final_hd95 = {}
    for name, metric in hd95_metrics.items():
        try:
            final_hd95[name] = f"{metric.aggregate().item():.4f} voxels"
        except Exception:
            final_hd95[name] = "N/A"
        metric.reset()

    print("\n==================================================")
    print("             FINAL TEST EVALUATION RESULTS        ")
    print("==================================================")
    print(f"  Prediction Threshold: {args.threshold}")
    print(f"  Dice (with background):        {final_dice['with_background']:.4f}")
    print(f"  Dice (lesion only):            {final_dice['lesion_only']:.4f}")
    print(f"  HD95 (with background):        {final_hd95['with_background']}")
    print(f"  HD95 (lesion only):            {final_hd95['lesion_only']}")
    print("  Empty prediction masks by original channel (HD95 warning cases):")
    print(f"    Background (original class 0): {empty_prediction_counts[0]}")
    print(f"    Lesion (original class 1):     {empty_prediction_counts[1]}")
    print("  Empty ground-truth masks by original channel (HD95 warning cases):")
    print(f"    Background (original class 0): {empty_label_counts[0]}")
    print(f"    Lesion (original class 1):     {empty_label_counts[1]}")
    lesion_prediction_cases = empty_prediction_counts[1]
    print("  MONAI warning-label mapping:")
    print(f"    with_background class 1 (lesion): {lesion_prediction_cases}")
    print(f"    lesion_only class 0 (lesion):     {lesion_prediction_cases}")
    print(f"  Slices with an empty prediction mask: {len(empty_prediction_slices)}")
    print(f"  Slices with an empty ground-truth mask: {len(empty_label_slices)}")
    if args.save_preds:
        print(f"  Predictions saved to: '{output_predictions_dir}'")
    else:
        print("  Prediction saving: DISABLED")
    print("==================================================\n")

if __name__ == "__main__":
    main()