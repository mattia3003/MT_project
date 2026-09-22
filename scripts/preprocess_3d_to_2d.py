import os
import glob
import nibabel as nib
import numpy as np

def extract_2d_slices_by_patient(data_dir, output_dir, axis=2):
    """
    Iterates through patient folders, extracts non-empty 2D slices 
    from paired BRAVO and SEG NIfTI files, and saves them into separate 
    'bravo' and 'seg' subfolders inside each patient's output directory.
    
    axis: 0 = Sagittal, 1 = Coronal, 2 = Axial
    """
    # 1. Find all patient subdirectories (e.g., METS_01, METS_04)
    patient_folders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    
    total_slices_extracted = 0
    
    for patient_path in patient_folders:
        patient_id = os.path.basename(patient_path)
        
        # 2. Locate the BRAVO and SEG files within the patient's folder
        bravo_files = glob.glob(os.path.join(patient_path, "*bravo*.nii*"))
        seg_files = glob.glob(os.path.join(patient_path, "*seg*.nii*"))
        
        if not bravo_files or not seg_files:
            print(f"Skipping {patient_id}: Missing BRAVO or SEG file.")
            continue
            
        img_path = bravo_files[0]
        mask_path = seg_files[0]
        
        # 3. Create 'bravo' and 'seg' subfolders inside the patient's output directory
        patient_bravo_dir = os.path.join(output_dir, patient_id, "bravo")
        patient_seg_dir = os.path.join(output_dir, patient_id, "seg")
        
        os.makedirs(patient_bravo_dir, exist_ok=True)
        os.makedirs(patient_seg_dir, exist_ok=True)
        
        # 4. Load 3D NIfTI volumes
        img_obj = nib.load(img_path)
        mask_obj = nib.load(mask_path)
        
        img_data = img_obj.get_fdata()
        mask_data = mask_obj.get_fdata()
        
        patient_slice_count = 0
        num_slices = img_data.shape[axis]
        
        # 5. Extract slices along chosen axis
        for idx in range(num_slices):
            if axis == 0:
                img_slice, mask_slice = img_data[idx, :, :], mask_data[idx, :, :]
            elif axis == 1:
                img_slice, mask_slice = img_data[:, idx, :], mask_data[:, idx, :]
            else:
                img_slice, mask_slice = img_data[:, :, idx], mask_data[:, :, idx]
            
            # Keep only slices that contain segmentation annotations (non-empty background)
            if np.max(mask_slice) > 0:
                out_img_name = f"{patient_id}_slice{idx+1:03d}_bravo.nii.gz"
                out_mask_name = f"{patient_id}_slice{idx+1:03d}_seg.nii.gz"
                
                # Save 2D slices as compressed NIfTI files into their respective subfolders
                nib.save(nib.Nifti1Image(img_slice, img_obj.affine), os.path.join(patient_bravo_dir, out_img_name))
                nib.save(nib.Nifti1Image(mask_slice, mask_obj.affine), os.path.join(patient_seg_dir, out_mask_name))
                
                patient_slice_count += 1
                
        print(f"Processed {patient_id}: Extracted {patient_slice_count} non-empty slice pairs.")
        total_slices_extracted += patient_slice_count

    print(f"\nPreprocessing Complete! Total extracted slices across all patients: {total_slices_extracted}")

# Usage:
if __name__ == "__main__":
    raw_data_dir = "data/StanfordSkullStripped_1mm/train" 
    output_sliced_dir = "processed_data/StanfordSkullStripped_1mm/train"
    
    extract_2d_slices_by_patient(data_dir=raw_data_dir, output_dir=output_sliced_dir, axis=2)