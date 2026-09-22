import os
import glob
import random
import argparse
import torch
import nibabel as nib
from pathlib import Path
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, MapTransform,
    ResizeWithPadOrCropd, NormalizeIntensityd, RandRotated,
    RandFlipd, RandAffined, RandGaussianNoised, RandAdjustContrastd, EnsureTyped
)
from monai.data import Dataset, CacheDataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

# ==============================================================================
# HELPER: SYNTHETIC NIFTI LOADER (FIXED DIMENSIONS)
# ==============================================================================
class LoadSyntheticNiftid(MapTransform):
    """
    Directly loads synthetic 2-slice NIfTI files (slice 0: image, slice 1: mask)
    into standard MONAI 2D tensors [1, H, W], ensuring correct spatial dimensions.
    """
    def __init__(self, keys=["data_file"]):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            filepath = d.pop(key)
            
            # Read array directly with Nibabel
            arr = nib.load(filepath).get_fdata()
            tensor = torch.from_numpy(arr).float()
            
            # Fix spatial shape: NIfTI typically loads as [H, W, 2] or [H, W, 1, 2]
            # Squeeze out extra 1-sized dimensions, keeping spatial dimensions intact
            tensor = tensor.squeeze()
            
            # If channels are last [H, W, 2], transpose to [2, H, W]
            if tensor.ndim == 3 and tensor.shape[-1] == 2:
                tensor = tensor.permute(2, 0, 1)
            elif tensor.ndim == 3 and tensor.shape[0] != 2:
                # Fallback check if shapes are unexpected
                raise ValueError(f"Unexpected synthetic tensor shape: {tensor.shape} from {filepath}")
            
            # Extract Image [1, H, W] and Mask [1, H, W]
            d["image"] = tensor[0:1, ...]
            d["label"] = (tensor[1:2, ...] > 0.5).float()
            
        return d

# ==============================================================================
# 1. UNIFIED DATASET BUILDER
# ==============================================================================
def get_dataloaders(dataset_type, data_dir, target_size=(256, 256), batch_size=16, seed=42):
    random.seed(seed)

    base_path = Path(data_dir).resolve()
    print(f"Resolving dataset directory: {base_path}")

    if not base_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {base_path}")

    # --------------------------------------------------------------------------
    # CASE A: ORIGINAL DATASET (Separate Folders: bravo/ and seg/)
    # --------------------------------------------------------------------------
    if dataset_type == "real":
        bravo_files = sorted([str(p) for p in base_path.glob("**/bravo/*.nii*")])
        seg_files = sorted([str(p) for p in base_path.glob("**/seg/*.nii*")])
        
        print(f"Found {len(bravo_files)} BRAVO files and {len(seg_files)} SEG files.")

        if len(bravo_files) == 0 or len(seg_files) == 0:
            raise FileNotFoundError(f"No NIfTI files found under '{base_path}'.")
            
        if len(bravo_files) != len(seg_files):
            raise ValueError(f"Mismatch in real dataset counts: {len(bravo_files)} BRAVO vs {len(seg_files)} SEG.")
        
        data_dicts = [{"image": b, "label": s} for b, s in zip(bravo_files, seg_files)]
        random.shuffle(data_dicts)

        load_transforms = [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]

    # --------------------------------------------------------------------------
    # CASE B: SYNTHETIC DATASET (Single File, 2-Slice Input)
    # --------------------------------------------------------------------------
    elif dataset_type == "synthetic":
        all_files = sorted([str(p) for p in base_path.glob("**/*.nii*")])
        print(f"Found {len(all_files)} synthetic files.")

        if len(all_files) == 0:
            raise FileNotFoundError(f"No synthetic .nii files found under '{base_path}'")

        data_dicts = [{"data_file": f} for f in all_files]
        random.shuffle(data_dicts)

        load_transforms = [
            LoadSyntheticNiftid(keys=["data_file"])
        ]

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}.")

    # --------------------------------------------------------------------------
    # SHARED TRANSFORMS (Preprocessing & Augmentations)
    # --------------------------------------------------------------------------
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

    # 80/20 Train/Val Split
    split_idx = int(0.8 * len(data_dicts))
    train_files = data_dicts[:split_idx]
    val_files = data_dicts[split_idx:]

    print(f"Loaded '{dataset_type}' dataset: {len(train_files)} train samples | {len(val_files)} val samples")

    # Fast Memory Caching for Synthetic Dataset to eliminate disk I/O bottleneck
    if dataset_type == "synthetic":
        print("Caching synthetic dataset in RAM for maximum speed...")
        train_ds = CacheDataset(data=train_files, transform=common_train_transforms, cache_rate=1.0, num_workers=4)
        val_ds = CacheDataset(data=val_files, transform=common_val_transforms, cache_rate=1.0, num_workers=2)
    else:
        train_ds = Dataset(data=train_files, transform=common_train_transforms)
        val_ds = Dataset(data=val_files, transform=common_val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader

# ==============================================================================
# 2. MAIN TRAINING ENGINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified 2D UNet Medical Image Segmentation")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["real", "synthetic"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_loader, val_loader = get_dataloaders(
        dataset_type=args.dataset_type,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    # SANITY CHECK: Print batch dimensions before starting training
    check_batch = next(iter(train_loader))
    print(f"\n--- DATA SANITY CHECK ---")
    print(f"Batch Image Shape: {check_batch['image'].shape} (Expected: [{args.batch_size}, 1, 256, 256])")
    print(f"Batch Label Shape: {check_batch['label'].shape} (Expected: [{args.batch_size}, 1, 256, 256])")
    print(f"Label Unique Values: {torch.unique(check_batch['label'])}")
    print(f"-------------------------\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)
    ).to(device)

    loss_function = DiceCELoss(sigmoid=True, squared_pred=True, lambda_dice=1.0, lambda_ce=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    best_metric = -1.0
    best_epoch = -1
    save_filename = f"best_{args.dataset_type}_model.pth"

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