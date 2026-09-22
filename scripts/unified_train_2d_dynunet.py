import random
import argparse
import torch
from pathlib import Path

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ResizeWithPadOrCropd, NormalizeIntensityd, RandRotated,
    RandFlipd, RandAffined, RandGaussianNoised, RandAdjustContrastd, EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import DynUNet
from monai.losses import DiceCELoss, DeepSupervisionLoss
from monai.metrics import DiceMetric

# ==============================================================================
# 1. UNIFIED DATASET BUILDER
# ==============================================================================
def get_dataloaders(data_dir, target_size=(256, 256), batch_size=32, seed=42):
    random.seed(seed)
    base_path = Path(data_dir).resolve()
    print(f"Resolving dataset directory: {base_path}")

    if not base_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {base_path}")

    # Finds all files inside /bravo and /seg subfolders
    bravo_files = sorted([str(p) for p in base_path.glob("**/bravo/*.nii*")])
    seg_files = sorted([str(p) for p in base_path.glob("**/seg/*.nii*")])

    print(f"Found {len(bravo_files)} BRAVO files and {len(seg_files)} SEG files.")

    if len(bravo_files) == 0 or len(seg_files) == 0:
        raise FileNotFoundError(f"No NIfTI files found in '{base_path}/bravo/' or '{base_path}/seg/'.")

    if len(bravo_files) != len(seg_files):
        raise ValueError(f"File count mismatch: {len(bravo_files)} BRAVO vs {len(seg_files)} SEG files.")

    data_dicts = [{"image": b, "label": s} for b, s in zip(bravo_files, seg_files)]
    random.shuffle(data_dicts)

    # Standard MONAI Loaders
    load_transforms = [
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
    ]

    # Preprocessing & Data Augmentation
    common_train_transforms = Compose([
        *load_transforms,
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=target_size, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        
        # Augmentations
        RandRotated(keys=["image", "label"], range_x=0.26, prob=0.5, mode=["bilinear", "nearest"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandAffined(keys=["image", "label"], prob=0.3, scale_range=(0.1, 0.1), mode=["bilinear", "nearest"]),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
        
        EnsureTyped(keys=["image", "label"])
    ])

    common_val_transforms = Compose([
        *load_transforms,
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=target_size, mode="constant"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"])
    ])

    # 80/20 Train/Validation Split
    split_idx = int(0.8 * len(data_dicts))
    train_files = data_dicts[:split_idx]
    val_files = data_dicts[split_idx:]

    print(f"Dataset split: {len(train_files)} train samples | {len(val_files)} val samples")

    # Fast standard PyTorch dataset reading directly from converted 2D files
    train_ds = Dataset(data=train_files, transform=common_train_transforms)
    val_ds = Dataset(data=val_files, transform=common_val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader

# ==============================================================================
# 2. MAIN TRAINING ENGINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified 2D DynUNet Medical Image Segmentation")
    parser.add_argument("--dataset_name", type=str, required=True, help="Label for saving checkpoints (e.g. real or synthetic)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the folder containing bravo/ and seg/ subfolders")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    # Sanity check execution
    check_batch = next(iter(train_loader))
    print(f"\n--- DATA SANITY CHECK ---")
    print(f"Batch Image Shape: {check_batch['image'].shape} (Expected: [{args.batch_size}, 1, 256, 256])")
    print(f"Batch Label Shape: {check_batch['label'].shape} (Expected: [{args.batch_size}, 1, 256, 256])")
    print(f"Label Unique Values: {torch.unique(check_batch['label'])}")
    print(f"-------------------------\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    base_loss = DiceCELoss(sigmoid=True, squared_pred=True, lambda_dice=1.0, lambda_ce=0.2)
    loss_function = DeepSupervisionLoss(base_loss, weights=[1.0, 0.5, 0.25, 0.125])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    best_metric = -1.0
    best_epoch = -1
    save_filename = f"best_{args.dataset_name}_model.pth"

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)

            if isinstance(outputs, torch.Tensor) and outputs.ndim == 5:
                outputs = torch.unbind(outputs, dim=1)
            elif isinstance(outputs, (list, tuple)):
                outputs = list(outputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        epoch_loss /= step

        # Validation Loop
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = model(val_images)

                if isinstance(val_outputs, (list, tuple)):
                    val_outputs = val_outputs[0]
                elif isinstance(val_outputs, torch.Tensor) and val_outputs.ndim == 5:
                    val_outputs = val_outputs[:, 0]

                val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()

            print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {epoch_loss:.4f} | Val Dice: {metric:.4f}")

            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch + 1
                torch.save(model.state_dict(), save_filename)
                print(f"   ---> Saved new best checkpoint: {save_filename}")

        scheduler.step()

    print(f"\nTraining Complete! Best Validation Dice: {best_metric:.4f} at Epoch {best_epoch}")

if __name__ == "__main__":
    main()