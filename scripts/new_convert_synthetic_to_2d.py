import argparse
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
- Exclusion: Optional command-line flag to filter out corrupted files listed in a .txt file.
"""

# Default paths
DEFAULT_SRC_DIR = Path("../data/Synthetic_data/mask_conditioned_synthesis")
DEFAULT_DST_DIR = Path("../processed_data/Synthetic_data_2d_10_0")
DEFAULT_EXCLUSION_FILE = Path("corrupted_synthetic_masks_10_0.txt")


def load_excluded_filenames(exclusion_path: Path) -> set:
    """Parses exclusion text file and returns a set of filenames/stems to ignore."""
    excluded = set()
    if not exclusion_path or not exclusion_path.exists():
        print(f"Warning: Exclusion file '{exclusion_path}' not found. No files will be excluded.")
        return excluded

    with open(exclusion_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comment headers
            if not line or line.startswith("#"):
                continue

            # Parse line format: "10004.seg.nii.gz | Failure Reason..."
            parts = line.split("|")
            filename = parts[0].strip()

            # Normalize name to match file stem/name (e.g. "10004.seg.nii.gz" -> "10004")
            raw_stem = filename.split(".")[0]
            excluded.add(raw_stem)
            excluded.add(filename)

    return excluded


def convert_synthetic_to_2d(src_dir: Path, dst_dir: Path, use_exclusion: bool, exclusion_file: Path):
    bravo_dir = dst_dir / "bravo"
    seg_dir = dst_dir / "seg"
    bravo_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Handle exclusion toggling
    excluded_set = set()
    if use_exclusion:
        excluded_set = load_excluded_filenames(exclusion_file)
        print(f"Exclusion filter ACTIVE: Loaded {len(excluded_set)} entry rules from '{exclusion_file}'")
    else:
        print("Exclusion filter INACTIVE: Converting ALL files in source directory.")

    all_synthetic_files = sorted(list(src_dir.glob("**/*.nii*")))
    
    # Filter out excluded files if toggled on
    synthetic_files = []
    skipped_count = 0
    for filepath in all_synthetic_files:
        stem = filepath.name.replace(".gz", "").replace(".nii", "")
        
        if use_exclusion and (filepath.name in excluded_set or stem in excluded_set or stem.split(".")[0] in excluded_set):
            skipped_count += 1
            continue
        synthetic_files.append(filepath)

    print(f"Found {len(all_synthetic_files)} total files.")
    if use_exclusion:
        print(f"Skipped {skipped_count} corrupted/excluded files.")
    print(f"Processing {len(synthetic_files)} valid files to convert...\n")

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
    parser = argparse.ArgumentParser(description="Convert Synthetic 3D NIfTI volumes to 2D slice folders.")
    
    # Toggle flag for exclusion
    parser.add_argument(
        "--exclude", 
        action="store_true", 
        help="Enable filtering out corrupted files listed in the exclusion text file."
    )
    parser.add_argument(
        "--exclusion_file", 
        type=str, 
        default=str(DEFAULT_EXCLUSION_FILE), 
        help="Path to the corrupted files .txt file (default: corrupted_files.txt)"
    )
    parser.add_argument(
        "--src_dir", 
        type=str, 
        default=str(DEFAULT_SRC_DIR), 
        help="Source directory containing synthetic NIfTI files."
    )
    parser.add_argument(
        "--dst_dir", 
        type=str, 
        default=str(DEFAULT_DST_DIR), 
        help="Destination directory for output 2D slice folders."
    )

    args = parser.parse_args()

    convert_synthetic_to_2d(
        src_dir=Path(args.src_dir),
        dst_dir=Path(args.dst_dir),
        use_exclusion=args.exclude,
        exclusion_file=Path(args.exclusion_file)
    )