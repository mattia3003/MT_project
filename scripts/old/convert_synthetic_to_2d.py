from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm

"""
Converts synthetic 3D NIfTI files into 2D slices for training a 2D UNet model.
- Input: 3D NIfTI files with shape [H, W, 2] where the last dimension contains [image, mask].
- Output: Two separate folders:
  - /bravo: Contains 2D image slices saved as NIfTI files.
  - /seg: Contains corresponding 2D mask slices saved as NIfTI files.
"""

# Update these paths to match your system
SRC_DIR = Path("../data/Synthetic_data/mask_conditioned_synthesis")  # Path to the folder containing synthetic NIfTI files
DST_DIR = Path("../processed_data/Synthetic_data_2d")  # Path to the output folder for 2D slices

def convert_synthetic_to_2d():
    bravo_dir = DST_DIR / "bravo"
    seg_dir = DST_DIR / "seg"
    bravo_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    synthetic_files = sorted(list(SRC_DIR.glob("**/*.nii*")))
    print(f"Found {len(synthetic_files)} files to convert...")

    for filepath in tqdm(synthetic_files):
        img_obj = nib.load(filepath)
        arr = img_obj.get_fdata().squeeze()

        # Fix spatial orientation: ensure shape is [2, H, W]
        if arr.ndim == 3 and arr.shape[-1] == 2:
            arr = np.transpose(arr, (2, 0, 1))

        image_slice = arr[0].astype(np.float32)
        mask_slice = (arr[1] > 0.5).astype(np.float32)

        # Save image slice to /bravo and mask slice to /seg
        stem = filepath.name.replace(".gz", "").replace(".nii", "")
        
        nib.save(nib.Nifti1Image(image_slice, img_obj.affine), bravo_dir / f"{stem}.bravo.nii.gz")
        nib.save(nib.Nifti1Image(mask_slice, img_obj.affine), seg_dir / f"{stem}.seg.nii.gz")

    print(f"\nSuccessfully converted {len(synthetic_files)} files to 2D!")
    print(f"Images saved to: {bravo_dir}")
    print(f"Masks saved to:  {seg_dir}")

if __name__ == "__main__":
    convert_synthetic_to_2d()