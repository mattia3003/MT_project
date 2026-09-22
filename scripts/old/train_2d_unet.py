import os
import glob
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    RandRotated,
    RandFlipd,
    EnsureTyped,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

def main():
    # --------------------------------------------------------------------------
    # 1. PATH DEFINITION & PATIENT-LEVEL TRAIN/VAL SPLIT
    # --------------------------------------------------------------------------
    # Path to the sliced training dataset root directory
    train_data_root = "processed_data/StanfordSkullStripped_1mm/train"  # Adjust to your folder path
    
    # Identify all patient subdirectories within the training folder only
    patient_dirs = sorted([
        d.path for d in os.scandir(train_data_root) if d.is_dir()
    ])
    
    if not patient_dirs:
        raise FileNotFoundError(f"No patient folders found in {train_data_root}")
        
    # Split patient folders (80% train, 20% validation) at the PATIENT level
    num_train_patients = int(0.8 * len(patient_dirs))
    train_patient_dirs = patient_dirs[:num_train_patients]
    val_patient_dirs = patient_dirs[num_train_patients:]

    def collect_slice_files(patient_dir_list):
        files = []
        for p_dir in patient_dir_list:
            images = sorted(glob.glob(os.path.join(p_dir, "bravo", "*.nii.gz")))
            masks = sorted(glob.glob(os.path.join(p_dir, "seg", "*.nii.gz")))
            for img, seg in zip(images, masks):
                files.append({"image": img, "label": seg})
        return files

    train_files = collect_slice_files(train_patient_dirs)
    val_files = collect_slice_files(val_patient_dirs)

    print(f"Patient Split: {len(train_patient_dirs)} train patients | {len(val_patient_dirs)} val patients")
    print(f"Slice Counts:  {len(train_files)} train slices   | {len(val_files)} val slices")

    # --------------------------------------------------------------------------
    # 2. MONAI TRANSFORM PIPELINES (DICTIONARY-BASED)
    # --------------------------------------------------------------------------
    TARGET_SIZE = (256, 256)  # Target size for resizing slices (height, width)
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=TARGET_SIZE, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandRotated(
            keys=["image", "label"],
            range_x=0.26,  # ~15 degrees in radians
            prob=0.5,
            mode=["bilinear", "nearest"]
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandAffined(
            keys=["image", "label"],
            prob=0.3,
            scale_range=(0.1, 0.1),
            mode=["bilinear", "nearest"]
        ),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
        EnsureTyped(keys=["image", "label"])
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=TARGET_SIZE, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"])
    ])

    # --------------------------------------------------------------------------
    # 3. DATASETS AND DATALOADERS
    # --------------------------------------------------------------------------
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)

    # --------------------------------------------------------------------------
    # 4. MODEL, LOSS FUNCTION, OPTIMIZER, & METRIC
    # --------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    max_epochs = 40

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    ).to(device)

    loss_function = DiceCELoss(sigmoid=True, squared_pred=True, lambda_dice=1.0, lambda_ce=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # --------------------------------------------------------------------------
    # 5. TRAINING AND VALIDATION LOOP
    # --------------------------------------------------------------------------
    best_metric = -1.0
    best_epoch = -1

    for epoch in range(max_epochs):
        print(f"\n--- Epoch {epoch + 1}/{max_epochs} ---")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= step
        print(f"Epoch {epoch + 1} Training Loss: {epoch_loss:.4f}")

        # Validation Phase
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = model(val_images)
                
                # Threshold model outputs to binary predictions
                val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()

            print(f"Epoch {epoch + 1} Validation Dice Score: {metric:.4f}")

            # Save model weights if validation metric improves
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model.pth")
                print(f"---> Saved new best model (Validation Dice: {best_metric:.4f})")

        scheduler.step()
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\nTraining Complete! Best Validation Dice: {best_metric:.4f} at Epoch {best_epoch}")

if __name__ == "__main__":
    main()