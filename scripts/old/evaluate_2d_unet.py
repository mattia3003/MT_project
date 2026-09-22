import os
import glob
import torch
import monai
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
    EnsureTyped,
    SaveImaged
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric

def main():
    # --------------------------------------------------------------------------
    # 1. PATH DEFINITION & TEST DATASET ISOLATION
    # --------------------------------------------------------------------------
    # Target strictly the test folder (Never touched during training/validation)
    test_data_root = "processed_data/StanfordSkullStripped_1mm/test"  
    output_predictions_dir = "models/predictions"
    model_weights_path = "models/best_synthetic_model.pth"

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights file '{model_weights_path}' not found. Train the model first!")

    patient_dirs = sorted([
        d.path for d in os.scandir(test_data_root) if d.is_dir()
    ])

    if not patient_dirs:
        raise FileNotFoundError(f"No patient folders found in {test_data_root}")

    # Gather test slice file pairs
    test_files = []
    for p_dir in patient_dirs:
        images = sorted(glob.glob(os.path.join(p_dir, "bravo", "*.nii.gz")))
        masks = sorted(glob.glob(os.path.join(p_dir, "seg", "*.nii.gz")))
        for img, seg in zip(images, masks):
            test_files.append({"image": img, "label": seg})

    print(f"Test Dataset Loaded: {len(patient_dirs)} patients | {len(test_files)} total 2D slices")

    # --------------------------------------------------------------------------
    # 2. EVALUATION TRANSFORM PIPELINE
    # --------------------------------------------------------------------------
    TARGET_SIZE = (256, 256)  # Target size for resizing slices (height, width)

    # Only deterministic transforms — no data augmentations
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),  # Retain metadata for saving output NIfTIs
        EnsureChannelFirstd(keys=["image", "label"]),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=TARGET_SIZE, mode="constant"),
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

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    ).to(device)

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # Setup metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=95, reduction="mean")

    # --------------------------------------------------------------------------
    # 4. INFERENCE & METRIC EVALUATION LOOP
    # --------------------------------------------------------------------------
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

            # Pass through network
            outputs = model(test_inputs)

            # Apply Sigmoid and threshold at 0.5 to form binary segmentation mask
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Compute evaluation metrics
            dice_metric(y_pred=preds, y=test_labels)
            try:
                hd95_metric(y_pred=preds, y=test_labels)
            except Exception:
                # Passes if binary prediction/ground truth has 0 positive voxels on a slice
                pass

            # Attach prediction tensor back to dictionary metadata for saving
            batch_data["pred"] = preds
            batch_data["pred_meta_dict"] = batch_data["image_meta_dict"]  # Use original image metadata for saving

            batch_list = decollate_batch(batch_data)
            for item in batch_list:
                saver(item)

    # --------------------------------------------------------------------------
    # 5. FINAL REPORTING
    # --------------------------------------------------------------------------
    final_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    try:
        final_hd95 = hd95_metric.aggregate().item()
        hd95_metric.reset()
        hd95_str = f"{final_hd95:.4f} voxels"
    except Exception:
        hd95_str = "N/A"

    print("\n==================================================")
    print("             FINAL TEST EVALUATION RESULTS        ")
    print("==================================================")
    print(f"  Mean Sørensen–Dice Coefficient: {final_dice:.4f} ({final_dice*100:.2f}%)")
    print(f"  95th Percentile Hausdorff Distance: {hd95_str}")
    print(f"  Predictions saved to: '{output_predictions_dir}'")
    print("==================================================\n")

if __name__ == "__main__":
    main()