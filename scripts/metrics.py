#!/usr/bin/env python3
"""
Unified Dataset Quality & Distribution Distance Evaluator for Medical MRI Slices

Handles Dataset Layouts:
  - Nested Patient Hierarchy: root_dir / patient_id / bravo / *.nii.gz
  - Flat Modality Folders:   root_dir / bravo / *.nii.gz

Supported Metrics:
  - Inception Score (IS)
  - Standard FID (Inception-v3)
  - Medical FID (MONAI DenseNet)
  - CMMD (CLIP Maximum Mean Discrepancy)
  - SSIM & Multi-Scale SSIM (MS-SSIM)
  - 1-Nearest Neighbor Feature Accuracy (1-NN)

Author: Medical Imaging AI Pipeline
"""

import argparse
from pathlib import Path
from typing import Set, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# MONAI Imports
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
    EnsureTyped
)

# TorchMetrics Imports
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
)

# ------------------------------------------------------------------------------
# 1. DATASET & UTILITY HELPERS
# ------------------------------------------------------------------------------

def load_exclusion_list(exclusion_file: Optional[Path]) -> Set[str]:
    """Parses corrupted text list and returns set of filenames/stems to ignore."""
    excluded = set()
    if not exclusion_file or not exclusion_file.exists():
        return excluded

    with open(exclusion_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            filename = line.split("|")[0].strip()
            raw_stem = filename.split(".")[0]
            excluded.add(raw_stem)
            excluded.add(filename)
    return excluded


class NiftiSliceDataset(Dataset):
    """
    Flexible MONAI-powered dataset for 2D NIfTI slices.
    
    Routes specifically to target modality folders (e.g. 'bravo') whether structured as:
      - Flat:   data_dir / modality / *.nii.gz
      - Nested: data_dir / patient_id / modality / *.nii.gz
    """
    def __init__(
        self, 
        data_dir: Path, 
        exclusion_set: Set[str], 
        modality: str = "bravo", 
        target_size: Tuple[int, int] = (256, 256)
    ):
        self.data_dir = data_dir
        
        # 1. Find all NIfTI files inside target modality subfolders (ignoring 'seg')
        all_found = sorted(list(data_dir.glob(f"**/{modality}/*.nii*")))
        
        # 2. Filter out corrupted slices via exclusion set
        self.files = []
        for f in all_found:
            filename = f.name
            raw_stem = filename.replace(".gz", "").replace(".nii", "")
            
            # Match against exclusion identifiers
            if filename in exclusion_set or raw_stem in exclusion_set or raw_stem.split(".")[0] in exclusion_set:
                continue
            
            self.files.append({"image": str(f)})

        print(f"[{data_dir.name} - '{modality}'] Loaded {len(self.files)} valid slice files (Excluded: {len(all_found) - len(self.files)})")

        if len(self.files) == 0:
            raise RuntimeError(f"No valid .nii/.nii.gz files found in {data_dir} under modality subfolder '{modality}'.")

        # MONAI Image Loading & Preprocessing Pipeline
        self.transforms = Compose([
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=target_size, mode="constant"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"], dtype=torch.float32)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transforms(self.files[idx])


# ------------------------------------------------------------------------------
# 2. METRIC CALCULATOR ENGINE
# ------------------------------------------------------------------------------

class MetricEvaluator:
    def __init__(self, device: torch.device):
        self.device = device

    # --- A. INCEPTION SCORE (IS) ---
    def compute_inception_score(self, loader: DataLoader) -> Tuple[float, float]:
        """Calculates Inception Score on a single dataset."""
        print("\n[Evaluating Inception Score (IS)]...")
        is_metric = InceptionScore(feature="logits_unbiased", normalize=True).to(self.device)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="IS Progress"):
                imgs = batch["image"].to(self.device)
                imgs_3ch = imgs.repeat(1, 3, 1, 1)
                imgs_3ch = (imgs_3ch - imgs_3ch.min()) / (imgs_3ch.max() - imgs_3ch.min() + 1e-8)
                is_metric.update(imgs_3ch)

        mean_is, std_is = is_metric.compute()
        return mean_is.item(), std_is.item()

    # --- B. STANDARD & MEDICAL FID ---
    def compute_fid(self, real_loader: DataLoader, syn_loader: DataLoader, use_medical_backbone: bool = False) -> float:
        """Calculates Standard (Inception-v3) or Medical (MONAI DenseNet) FID."""
        metric_name = "Medical FID (DenseNet)" if use_medical_backbone else "Standard FID (Inception-v3)"
        print(f"\n[Evaluating {metric_name}]...")

        if not use_medical_backbone:
            fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(self.device)
            with torch.no_grad():
                for batch in tqdm(real_loader, desc="Processing Real Dataset"):
                    imgs = batch["image"].to(self.device).repeat(1, 3, 1, 1)
                    imgs_uint8 = ((imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8) * 255).to(torch.uint8)
                    fid.update(imgs_uint8, real=True)

                for batch in tqdm(syn_loader, desc="Processing Synthetic Dataset"):
                    imgs = batch["image"].to(self.device).repeat(1, 3, 1, 1)
                    imgs_uint8 = ((imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8) * 255).to(torch.uint8)
                    fid.update(imgs_uint8, real=False)

            return fid.compute().item()
        else:
            feature_extractor = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(self.device)
            feature_extractor.class_layers = nn.Identity()
            feature_extractor.eval()

            def extract_features(loader):
                feats = []
                with torch.no_grad():
                    for batch in tqdm(loader, desc="Extracting Medical Embeddings"):
                        imgs = batch["image"].to(self.device)
                        emb = feature_extractor(imgs)
                        feats.append(emb.cpu().numpy())
                return np.concatenate(feats, axis=0)

            real_feats = extract_features(real_loader)
            syn_feats = extract_features(syn_loader)

            mu_r, sigma_r = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
            mu_s, sigma_s = np.mean(syn_feats, axis=0), np.cov(syn_feats, rowvar=False)
            
            from scipy.linalg import sqrtm
            ssdiff = np.sum((mu_r - mu_s) ** 2.0)
            covmean = sqrtm(sigma_r.dot(sigma_s))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            med_fid = ssdiff + np.trace(sigma_r + sigma_s - 2.0 * covmean)
            return float(med_fid)

    # --- C. CMMD (CLIP MAXIMUM MEAN DISCREPANCY) ---
    def compute_cmmd(
        self, 
        real_loader: DataLoader, 
        syn_loader: DataLoader, 
        clip_model_name: str = "openai/clip-vit-base-patch32", 
        sigma: float = 10.0
    ) -> float:
        """Calculates CLIP Maximum Mean Discrepancy (CMMD) between Real and Synthetic datasets."""
        print(f"\n[Evaluating CMMD using CLIP ({clip_model_name})]...")
        try:
            from transformers import CLIPVisionModelWithProjection
        except ImportError:
            raise ImportError("Please install `transformers` via `pip install transformers` to compute CMMD.")

        model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(self.device)
        model.eval()

        def extract_clip_embeddings(loader):
            embeddings = []
            with torch.no_grad():
                for batch in tqdm(loader, desc="Extracting CLIP Features"):
                    imgs = batch["image"].to(self.device)
                    imgs_3ch = imgs.repeat(1, 3, 1, 1)
                    imgs_3ch = (imgs_3ch - imgs_3ch.min()) / (imgs_3ch.max() - imgs_3ch.min() + 1e-8)
                    
                    outputs = model(pixel_values=imgs_3ch)
                    proj_embeds = outputs.image_embeds
                    proj_embeds = proj_embeds / proj_embeds.norm(dim=-1, keepdim=True)
                    embeddings.append(proj_embeds.cpu())
                    
            return torch.cat(embeddings, dim=0)

        x = extract_clip_embeddings(real_loader).to(self.device)
        y = extract_clip_embeddings(syn_loader).to(self.device)

        def rbf_kernel(a: torch.Tensor, b: torch.Tensor, bandwidth: float):
            dist_sq = torch.cdist(a, b, p=2) ** 2
            return torch.exp(-dist_sq / (2.0 * bandwidth ** 2))

        n, m = x.shape[0], y.shape[0]
        k_xx = rbf_kernel(x, x, sigma)
        k_yy = rbf_kernel(y, y, sigma)
        k_xy = rbf_kernel(x, y, sigma)

        k_xx.fill_diagonal_(0.0)
        k_yy.fill_diagonal_(0.0)

        mmd_sq = (k_xx.sum() / (n * (n - 1))) + (k_yy.sum() / (m * (m - 1))) - (2.0 * k_xy.sum() / (n * m))
        return float(torch.clamp(mmd_sq, min=0.0).sqrt().item())

    # --- D. SSIM & MULTI-SCALE SSIM (MS-SSIM) ---
    def compute_ssim_metrics(self, real_loader: DataLoader, syn_loader: DataLoader) -> Tuple[float, float]:
        """Calculates mean SSIM and MS-SSIM across dataset distributions."""
        print("\n[Evaluating SSIM and Multi-Scale SSIM (MS-SSIM)]...")
        ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        ms_ssim_calc = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        ssim_scores, ms_ssim_scores = [], []
        
        with torch.no_grad():
            for batch_r, batch_s in zip(real_loader, syn_loader):
                imgs_r = batch_r["image"].to(self.device)
                imgs_s = batch_s["image"].to(self.device)
                
                min_b = min(imgs_r.shape[0], imgs_s.shape[0])
                imgs_r, imgs_s = imgs_r[:min_b], imgs_s[:min_b]

                imgs_r = (imgs_r - imgs_r.min()) / (imgs_r.max() - imgs_r.min() + 1e-8)
                imgs_s = (imgs_s - imgs_s.min()) / (imgs_s.max() - imgs_s.min() + 1e-8)

                ssim_val = ssim_calc(imgs_s, imgs_r)
                ms_ssim_val = ms_ssim_calc(imgs_s, imgs_r)

                ssim_scores.append(ssim_val.item())
                ms_ssim_scores.append(ms_ssim_val.item())

        return float(np.mean(ssim_scores)), float(np.mean(ms_ssim_scores))

    # --- E. 1-NEAREST NEIGHBOR (1-NN) FEATURE ACCURACY ---
    def compute_1nn_accuracy(self, real_loader: DataLoader, syn_loader: DataLoader) -> float:
        """Calculates 1-NN classification accuracy between Real (0) and Synthetic (1). Target = 50%."""
        print("\n[Evaluating 1-Nearest Neighbor Feature Distance]...")
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        feature_extractor = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(self.device)
        feature_extractor.class_layers = nn.Identity()
        feature_extractor.eval()

        real_feats, syn_feats = [], []
        with torch.no_grad():
            for batch in real_loader:
                real_feats.append(feature_extractor(batch["image"].to(self.device)).cpu().numpy())
            for batch in syn_loader:
                syn_feats.append(feature_extractor(batch["image"].to(self.device)).cpu().numpy())

        X_real = np.concatenate(real_feats, axis=0)
        X_syn = np.concatenate(syn_feats, axis=0)

        min_samples = min(len(X_real), len(X_syn))
        X = np.vstack([X_real[:min_samples], X_syn[:min_samples]])
        y = np.array([0] * min_samples + [1] * min_samples)

        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        knn.fit(X, y)
        preds = knn.predict(X)
        acc = accuracy_score(y, preds)
        
        return float(acc)


# ------------------------------------------------------------------------------
# 3. CLI ENTRY POINT & SUBCOMMAND ROUTING
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Quality & Metric Evaluator for Medical MRI Datasets")
    
    # Dataset Paths
    parser.add_argument("--real_dir", type=str, required=True, help="Path to Real root dir ('train' or 'test')")
    parser.add_argument("--syn_dir", type=str, default=None, help="Path to Synthetic root dir")
    parser.add_argument("--modality", type=str, default="bravo", help="Subfolder modality name to evaluate ('bravo')")
    parser.add_argument("--exclusion_file", type=str, default="corrupted_files.txt", help="Path to corrupted list .txt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for metric loader")
    
    # Metric Selection Flags
    parser.add_argument("--all", action="store_true", help="Run all available metrics")
    parser.add_argument("--is_score", action="store_true", help="Compute Inception Score")
    parser.add_argument("--fid", action="store_true", help="Compute Standard FID (Inception-v3)")
    parser.add_argument("--med_fid", action="store_true", help="Compute Medical FID (MONAI DenseNet)")
    parser.add_argument("--cmmd", action="store_true", help="Compute CMMD (CLIP-based MMD)")
    parser.add_argument("--ssim", action="store_true", help="Compute SSIM and MS-SSIM")
    parser.add_argument("--one_nn", action="store_true", help="Compute 1-Nearest Neighbor Accuracy")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Metric Evaluation Engine on Device: {device}")

    # Load Exclusion Rules
    exclusion_set = load_exclusion_list(Path(args.exclusion_file))

    # Instantiate Datasets
    real_ds = NiftiSliceDataset(Path(args.real_dir), exclusion_set=set(), modality=args.modality)
    real_loader = DataLoader(real_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    syn_loader = None
    if args.syn_dir:
        syn_ds = NiftiSliceDataset(Path(args.syn_dir), exclusion_set=exclusion_set, modality=args.modality)
        syn_loader = DataLoader(syn_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    evaluator = MetricEvaluator(device=device)

    # Output Reporting Table
    print("\n==================================================")
    print("             DATASET METRIC REPORT                ")
    print("==================================================")

    if args.all or args.is_score:
        mean_is, std_is = evaluator.compute_inception_score(real_loader)
        print(f"  Real Inception Score (IS):   {mean_is:.4f} +/- {std_is:.4f}")
        if syn_loader:
            s_mean_is, s_std_is = evaluator.compute_inception_score(syn_loader)
            print(f"  Syn Inception Score (IS):    {s_mean_is:.4f} +/- {s_std_is:.4f}")

    if syn_loader:
        if args.all or args.fid:
            std_fid = evaluator.compute_fid(real_loader, syn_loader, use_medical_backbone=False)
            print(f"  Standard FID (Inception-v3): {std_fid:.4f}")

        if args.all or args.med_fid:
            m_fid = evaluator.compute_fid(real_loader, syn_loader, use_medical_backbone=True)
            print(f"  Medical FID (MONAI DenseNet):{m_fid:.4f}")

        if args.all or args.cmmd:
            cmmd_val = evaluator.compute_cmmd(real_loader, syn_loader)
            print(f"  CMMD (CLIP-MMD):             {cmmd_val:.6f}")

        if args.all or args.ssim:
            ssim_v, ms_ssim_v = evaluator.compute_ssim_metrics(real_loader, syn_loader)
            print(f"  Mean SSIM:                   {ssim_v:.4f}")
            print(f"  Mean MS-SSIM:                {ms_ssim_v:.4f}")

        if args.all or args.one_nn:
            one_nn_acc = evaluator.compute_1nn_accuracy(real_loader, syn_loader)
            print(f"  1-NN Feature Accuracy:       {one_nn_acc * 100:.2f}% (Target: 50.0%)")

    print("==================================================\n")


if __name__ == "__main__":
    main()