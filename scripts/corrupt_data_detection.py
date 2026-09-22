import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import label

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
SYNTHETIC_DIR = "../processed_data/Synthetic_data_2d"  # Or your .npy directory
OUTPUT_CORRUPTED_TXT = "corrupted_synthetic_masks.txt"

# Thresholds for flagging corrupted "starry sky" masks
MAX_ALLOWED_COMPONENTS = 10   # Real 2D slices rarely have > 5 distinct lesion blobs
MAX_BACKGROUND_PIXELS = 0     # Positives outside skull boundary (if skull background == 0)

def load_array(filepath):
    """Loads either .nii.gz or .npy file into a 2D numpy array."""
    path_str = str(filepath)
    if path_str.endswith(".npy"):
        arr = np.load(path_str)
    else:
        arr = nib.load(path_str).get_fdata()
    return np.squeeze(arr).astype(np.float32)

def audit_dataset():
    base_path = Path(SYNTHETIC_DIR).resolve()
    bravo_files = sorted(list(base_path.glob("**/bravo/*.*")))
    seg_files = sorted(list(base_path.glob("**/seg/*.*")))

    if len(bravo_files) == 0 or len(seg_files) == 0:
        print(f"Error: No files found in {base_path}")
        return

    print(f"Auditing {len(seg_files)} synthetic segmentation slices...")

    corrupted_files = []

    for img_path, seg_path in tqdm(zip(bravo_files, seg_files), total=len(seg_files)):
        img = load_array(img_path)
        seg = load_array(seg_path)

        # Ensure binary threshold for mask
        binary_seg = (seg > 0.5).astype(np.uint8)
        
        # If the label is completely empty, it's fine (background slice)
        if np.sum(binary_seg) == 0:
            continue

        reasons = []

        # Check 1: Positive mask pixels outside the skull (Where BRAVO intensity is zero/background)
        # Assumes background MRI intensity is <= 0 or very close to zero
        background_mask = (img <= 0.01)
        pixels_outside_brain = np.sum(binary_seg[background_mask])

        if pixels_outside_brain > MAX_BACKGROUND_PIXELS:
            reasons.append(f"Pixels outside brain: {pixels_outside_brain}")

        # Check 2: "Starry Sky" test (Count distinct disconnected mask fragments)
        labeled_array, num_features = label(binary_seg)
        if num_features > MAX_ALLOWED_COMPONENTS:
            reasons.append(f"Excessive noise fragments: {num_features} distinct blobs")

        if reasons:
            corrupted_files.append((seg_path.name, ", ".join(reasons)))

    # --------------------------------------------------------------------------
    # REPORTING & SAVING RESULTS
    # --------------------------------------------------------------------------
    print("\n" + "="*60)
    print("           SYNTHETIC SEGMENTATION AUDIT RESULTS            ")
    print("="*60)
    print(f" Total Slices Audited : {len(seg_files)}")
    print(f" Corrupted Masks Found: {len(corrupted_files)} ({len(corrupted_files)/len(seg_files)*100:.2f}%)")
    print("="*60)

    # Save flagged list to text file
    with open(OUTPUT_CORRUPTED_TXT, "w") as f:
        f.write("# Corrupted Synthetic Segmentation Files\n")
        f.write("# Format: Filename | Failure Reason\n")
        for filename, reason in corrupted_files:
            f.write(f"{filename} | {reason}\n")

    print(f"\nFull list of corrupted filenames saved to: '{OUTPUT_CORRUPTED_TXT}'")

if __name__ == "__main__":
    audit_dataset()