#!/usr/bin/env python3
"""
Unified Dataset Quality & Distribution Distance Evaluator for Medical MRI Slices

Handles Dataset Layouts:
  - Nested Patient Hierarchy: root_dir / patient_id / bravo / *.nii.gz
  - Flat Modality Folders:   root_dir / bravo / *.nii.gz

Supported Metrics:
  - Inception Score (IS)
  - Standard FID (Inception-v3)
    - Medical FID (RadImageNet ResNet-50)
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
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet50

# MONAI Imports
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    Resized,
    ScaleIntensityd,
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

        Metric preprocessing intentionally uses min-max scaling to [0, 1]. This
        differs from the segmentation training pipeline, which uses nonzero
        z-score normalization; Inception Score and standard FID require image
        inputs in the [0, 1] or uint8 range.
    """
    def __init__(
        self, 
        data_dir: Path, 
        exclusion_set: Set[str], 
        modality: str = "bravo", 
        target_size: Tuple[int, int] = (256, 256),
        resize_mode: str = "pad_crop",
    ):
        self.data_dir = data_dir

        if resize_mode not in {"pad_crop", "resize"}:
            raise ValueError("resize_mode must be 'pad_crop' or 'resize'")
        
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
        spatial_transform = (
            ResizeWithPadOrCropd(keys=["image"], spatial_size=target_size, mode="constant")
            if resize_mode == "pad_crop"
            else Resized(keys=["image"], spatial_size=target_size, mode="bilinear")
        )
        self.transforms = Compose([
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            spatial_transform,
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

    def _sample_loader(
        self,
        loader: DataLoader,
        max_samples: Optional[int],
        seed: int,
    ) -> DataLoader:
        """Return a reproducible random subset without changing image transforms."""
        if max_samples is None or max_samples >= len(loader.dataset):
            return loader

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(loader.dataset), generator=generator)[:max_samples].tolist()
        return DataLoader(
            Subset(loader.dataset, indices),
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=0,
        )

    @staticmethod
    def _as_rgb_uint8(images: torch.Tensor) -> torch.Tensor:
        """Convert metric images in [0, 1] to the uint8 RGB format required by FID."""
        images = images.float().clamp(0.0, 1.0)
        if images.ndim != 4:
            raise ValueError(f"Expected image batches with shape [B, C, H, W], got {tuple(images.shape)}.")
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 image channels for FID, got {images.shape[1]}.")
        return (images * 255.0).round().to(torch.uint8)

    def _load_radimagenet_backbone(self, weights_path: Path) -> nn.Module:
        """Load the official RadImageNet ResNet-50 convolutional backbone."""
        if not weights_path.exists():
            raise FileNotFoundError(f"RadImageNet weights not found: {weights_path}")

        model = resnet50(weights=None)
        backbone = nn.Sequential(*list(model.children())[:9]).to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model", "backbone"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        if not isinstance(checkpoint, dict):
            raise ValueError("RadImageNet checkpoint must contain a PyTorch state dictionary.")

        cleaned_state = {}
        for key, value in checkpoint.items():
            key = key.removeprefix("module.").removeprefix("backbone.")
            if key.startswith("fc."):
                continue
            cleaned_state[key] = value

        load_result = backbone.load_state_dict(cleaned_state, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "RadImageNet checkpoint does not match a ResNet-50 backbone. "
                f"Missing keys: {load_result.missing_keys[:3]}, "
                f"unexpected keys: {load_result.unexpected_keys[:3]}"
            )
        backbone.eval()
        return backbone

    @staticmethod
    def _as_radimagenet_input(images: torch.Tensor) -> torch.Tensor:
        """Prepare images using the official RadImageNet input convention."""
        images = images.float().clamp(0.0, 1.0)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 image channels for RadImageNet, got {images.shape[1]}.")
        images = torch.nn.functional.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        return images * 2.0 - 1.0

    @staticmethod
    def _frechet_distance(features_real: np.ndarray, features_syn: np.ndarray) -> float:
        """Calculate a numerically stable Frechet distance between feature sets."""
        from scipy.linalg import sqrtm

        if len(features_real) < 2 or len(features_syn) < 2:
            raise ValueError("Medical FID requires at least two samples per dataset.")

        features_real = np.asarray(features_real, dtype=np.float64)
        features_syn = np.asarray(features_syn, dtype=np.float64)
        mean_real = np.mean(features_real, axis=0)
        mean_syn = np.mean(features_syn, axis=0)
        covariance_real = np.cov(features_real, rowvar=False)
        covariance_syn = np.cov(features_syn, rowvar=False)

        # With 400 samples and 2048 features, the sample covariance is
        # necessarily rank-deficient. A small, scale-aware diagonal jitter
        # makes the Gaussian square root well-defined without changing the
        # feature distribution materially.
        feature_dim = covariance_real.shape[0]
        scale = max(
            np.trace(covariance_real) / feature_dim,
            np.trace(covariance_syn) / feature_dim,
            1.0,
        )
        covariance_real += np.eye(feature_dim) * (1e-6 * scale)
        covariance_syn += np.eye(feature_dim) * (1e-6 * scale)

        covariance_product = covariance_real @ covariance_syn
        covariance_mean = sqrtm(covariance_product)
        if np.iscomplexobj(covariance_mean):
            covariance_mean = covariance_mean.real

        mean_difference = mean_real - mean_syn
        distance = (
            mean_difference @ mean_difference
            + np.trace(covariance_real + covariance_syn - 2.0 * covariance_mean)
        )
        return float(max(distance, 0.0))

    @staticmethod
    def _rbf_mmd(
        features_real: torch.Tensor,
        features_syn: torch.Tensor,
        sigma: Optional[float],
    ) -> Tuple[float, float]:
        """Calculate unbiased RBF MMD from two normalized feature matrices."""
        n, m = len(features_real), len(features_syn)
        if n < 2 or m < 2:
            raise ValueError("MMD requires at least two samples in each dataset.")

        distance_sample = torch.cdist(
            features_real[: min(n, 512)], features_syn[: min(m, 512)], p=2
        )
        bandwidth = float(torch.median(distance_sample).item()) if sigma is None else sigma
        if bandwidth <= 0.0:
            raise ValueError(f"MMD bandwidth must be positive, got {bandwidth}.")

        def rbf_sum(a: torch.Tensor, b: torch.Tensor, block_size: int = 256) -> float:
            total = 0.0
            for start_a in range(0, len(a), block_size):
                for start_b in range(0, len(b), block_size):
                    distances_sq = torch.cdist(
                        a[start_a:start_a + block_size],
                        b[start_b:start_b + block_size],
                        p=2,
                    ).square()
                    total += torch.exp(-distances_sq / (2.0 * bandwidth ** 2)).sum().item()
            return total

        mmd_sq = (
            (rbf_sum(features_real, features_real) - n) / (n * (n - 1))
            + (rbf_sum(features_syn, features_syn) - m) / (m * (m - 1))
            - 2.0 * rbf_sum(features_real, features_syn) / (n * m)
        )
        return float(np.sqrt(max(mmd_sq, 0.0))), bandwidth

    # --- A. INCEPTION SCORE (IS) ---
    def compute_inception_score(self, loader: DataLoader) -> Tuple[float, float]:
        """Calculate Inception Score for images already scaled to ``[0, 1]``.

        The dataset transform performs the intensity scaling. Re-scaling each
        slice here would give every slice a separate intensity mapping and
        would make the score depend on slice-specific extrema.
        """
        print("\n[Evaluating Inception Score (IS)]...")
        is_metric = InceptionScore(feature="logits_unbiased", normalize=True).to(self.device)

        with torch.no_grad():
            for batch in tqdm(loader, desc="IS Progress"):
                # Extract images (handles dictionary or tensor loaders)
                imgs = batch["image"] if isinstance(batch, dict) else batch[0]
                imgs = imgs.to(self.device, dtype=torch.float32)

                if imgs.ndim != 4:
                    raise ValueError(
                        f"Expected image batches with shape [B, C, H, W], got {tuple(imgs.shape)}."
                    )

                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                elif imgs.shape[1] != 3:
                    raise ValueError(
                        f"Expected 1 or 3 image channels for Inception Score, got {imgs.shape[1]}."
                    )

                is_metric.update(imgs.clamp(0.0, 1.0))

        mean_is, std_is = is_metric.compute()
        return mean_is.item(), std_is.item()

    # --- B. (b1) STANDARD & (b2) MEDICAL FID ---
    def compute_fid(
        self,
        real_loader: DataLoader,
        syn_loader: DataLoader,
        use_medical_backbone: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42,
        medical_weights: Optional[Path] = None,
    ) -> float:
        """Calculate FID using equal, reproducible sample counts when requested.

        ``max_samples=400`` reproduces a capped comparison such as the one
        used by the earlier thesis. Leaving it as ``None`` uses all available
        samples, which is preferable for a final estimate when both datasets
        are sufficiently large.
        """
        metric_name = "Medical FID (RadImageNet ResNet-50)" if use_medical_backbone else "Standard FID (Inception-v3)"
        print(f"\n[Evaluating {metric_name}]...")

        real_loader = self._sample_loader(real_loader, max_samples, seed)
        syn_loader = self._sample_loader(syn_loader, max_samples, seed + 1)
        print(
            f"FID sample size: real={len(real_loader.dataset)}, "
            f"synthetic={len(syn_loader.dataset)}; seeds: real={seed}, synthetic={seed + 1}"
        )

        if not use_medical_backbone:
            fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(self.device)
            with torch.no_grad():
                for batch in tqdm(real_loader, desc="Processing Real Dataset"):
                    images = self._as_rgb_uint8(batch["image"].to(self.device))
                    fid.update(images, real=True)

                for batch in tqdm(syn_loader, desc="Processing Synthetic Dataset"):
                    images = self._as_rgb_uint8(batch["image"].to(self.device))
                    fid.update(images, real=False)

            return fid.compute().item()
        else:
            if medical_weights is None:
                raise ValueError(
                    "Medical FID requires a RadImageNet ResNet-50 checkpoint. "
                    "Pass it with --med_fid_weights."
                )
            feature_extractor = self._load_radimagenet_backbone(medical_weights)

            def extract_features(loader):
                feats = []
                with torch.no_grad():
                    for batch in tqdm(loader, desc="Extracting Medical Embeddings"):
                        images = self._as_radimagenet_input(batch["image"].to(self.device))
                        embeddings = feature_extractor(images).flatten(1)
                        feats.append(embeddings.cpu().numpy())
                return np.concatenate(feats, axis=0)

            real_feats = extract_features(real_loader)
            syn_feats = extract_features(syn_loader)

            return self._frechet_distance(real_feats, syn_feats)

    # --- C. CMMD (CLIP MAXIMUM MEAN DISCREPANCY) ---
    def compute_cmmd(
        self, 
        real_loader: DataLoader, 
        syn_loader: DataLoader, 
        clip_model_name: str = "openai/clip-vit-base-patch32", 
        sigma: Optional[float] = None,
        kernel_batch_size: int = 256,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> float:
        """Calculate CLIP-feature MMD with an RBF kernel.

        By default, the kernel bandwidth is the median cross-dataset feature
        distance. Kernel sums are evaluated in blocks so memory use does not
        grow quadratically with the number of slices.
        """
        print(f"\n[Evaluating CMMD using CLIP ({clip_model_name})]...")
        real_loader = self._sample_loader(real_loader, max_samples, seed)
        syn_loader = self._sample_loader(syn_loader, max_samples, seed + 1)
        print(
            f"CMMD sample size: real={len(real_loader.dataset)}, "
            f"synthetic={len(syn_loader.dataset)}; seeds: real={seed}, synthetic={seed + 1}"
        )
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        except ImportError:
            raise ImportError("Please install `transformers` via `py -3 -m pip install transformers` to compute CMMD.")

        processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(self.device)
        model.eval()

        image_size = processor.crop_size.get("height", 224)
        image_mean = torch.tensor(processor.image_mean, device=self.device).view(1, 3, 1, 1)
        image_std = torch.tensor(processor.image_std, device=self.device).view(1, 3, 1, 1)

        def extract_clip_embeddings(loader):
            embeddings = []
            with torch.no_grad():
                for batch in tqdm(loader, desc="Extracting CLIP Features"):
                    images = batch["image"].to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
                    if images.shape[1] == 1:
                        images = images.repeat(1, 3, 1, 1)
                    elif images.shape[1] != 3:
                        raise ValueError(f"Expected 1 or 3 image channels for CMMD, got {images.shape[1]}.")
                    images = torch.nn.functional.interpolate(
                        images,
                        size=(image_size, image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    images = (images - image_mean) / image_std

                    outputs = model(pixel_values=images)
                    proj_embeds = outputs.image_embeds
                    proj_embeds = proj_embeds / proj_embeds.norm(dim=-1, keepdim=True)
                    embeddings.append(proj_embeds.cpu())
                    
            return torch.cat(embeddings, dim=0)

        x = extract_clip_embeddings(real_loader)
        y = extract_clip_embeddings(syn_loader)
        n, m = x.shape[0], y.shape[0]
        if n < 2 or m < 2:
            raise ValueError("CMMD requires at least two samples in each dataset.")

        distance_sample = torch.cdist(x[: min(n, 512)], y[: min(m, 512)], p=2)
        bandwidth = float(torch.median(distance_sample).item()) if sigma is None else sigma
        if bandwidth <= 0.0:
            raise ValueError(f"CMMD bandwidth must be positive, got {bandwidth}.")
        print(f"CMMD kernel bandwidth (sigma): {bandwidth:.6f}")

        def rbf_sum(a: torch.Tensor, b: torch.Tensor) -> float:
            total = 0.0
            for start_a in range(0, len(a), kernel_batch_size):
                block_a = a[start_a:start_a + kernel_batch_size]
                for start_b in range(0, len(b), kernel_batch_size):
                    block_b = b[start_b:start_b + kernel_batch_size]
                    distances_sq = torch.cdist(block_a, block_b, p=2).square()
                    total += torch.exp(-distances_sq / (2.0 * bandwidth ** 2)).sum().item()
            return total

        k_xx = rbf_sum(x, x) - n
        k_yy = rbf_sum(y, y) - m
        k_xy = rbf_sum(x, y)
        mmd_sq = k_xx / (n * (n - 1)) + k_yy / (m * (m - 1)) - (2.0 * k_xy / (n * m))
        return float(np.sqrt(max(mmd_sq, 0.0)))

    # --- D. RADIMAGENET MAXIMUM MEAN DISCREPANCY ---
    def compute_radimagenet_mmd(
        self,
        real_loader: DataLoader,
        syn_loader: DataLoader,
        medical_weights: Optional[Path] = None,
        sigma: Optional[float] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> float:
        """Calculate RBF-MMD in the frozen RadImageNet feature space."""
        print("\n[Evaluating RadImageNet-MMD]")
        if medical_weights is None:
            raise ValueError(
                "RadImageNet-MMD requires a RadImageNet ResNet-50 checkpoint. "
                "Pass it with --med_fid_weights."
            )

        real_loader = self._sample_loader(real_loader, max_samples, seed)
        syn_loader = self._sample_loader(syn_loader, max_samples, seed + 1)
        print(
            f"RadImageNet-MMD sample size: real={len(real_loader.dataset)}, "
            f"synthetic={len(syn_loader.dataset)}; seeds: real={seed}, synthetic={seed + 1}"
        )
        feature_extractor = self._load_radimagenet_backbone(medical_weights)

        def extract_features(loader: DataLoader) -> torch.Tensor:
            features = []
            with torch.no_grad():
                for batch in tqdm(loader, desc="Extracting RadImageNet Features"):
                    images = self._as_radimagenet_input(batch["image"].to(self.device))
                    embeddings = feature_extractor(images).flatten(1)
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    features.append(embeddings.cpu())
            return torch.cat(features, dim=0)

        real_features = extract_features(real_loader)
        syn_features = extract_features(syn_loader)
        value, bandwidth = self._rbf_mmd(real_features, syn_features, sigma)
        print(f"RadImageNet-MMD kernel bandwidth (sigma): {bandwidth:.6f}")
        return value

    # --- E. (e1) SSIM & (e2) MULTI-SCALE SSIM (MS-SSIM) ---
    def compute_ssim_metrics(
        self,
        real_loader: DataLoader,
        syn_loader: DataLoader,
        paired: bool = False,
    ) -> Tuple[float, float]:
        """Calculate SSIM and MS-SSIM for explicitly paired images.

        SSIM compares corresponding pixels, so arbitrary real/synthetic
        pairing is not a valid distribution-level metric. The caller must
        explicitly opt into paired evaluation when both loaders contain the
        same number of spatially aligned images in the same order.
        """
        print("\n[Evaluating SSIM and Multi-Scale SSIM (MS-SSIM)]...")
        if not paired:
            raise ValueError(
                "SSIM/MS-SSIM require paired, spatially aligned images. "
                "The current real and synthetic datasets are unpaired; "
                "provide a paired loader or omit --ssim."
            )
        if len(real_loader.dataset) != len(syn_loader.dataset):
            raise ValueError(
                "Paired SSIM/MS-SSIM requires equal numbers of real and synthetic images."
            )

        ssim_calc = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        ms_ssim_calc = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        ssim_scores, ms_ssim_scores = [], []

        with torch.no_grad():
            for batch_r, batch_s in zip(real_loader, syn_loader):
                imgs_r = batch_r["image"].to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
                imgs_s = batch_s["image"].to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
                if imgs_r.shape != imgs_s.shape:
                    raise ValueError(
                        f"Paired SSIM/MS-SSIM requires matching image shapes, "
                        f"got {tuple(imgs_r.shape)} and {tuple(imgs_s.shape)}."
                    )

                ssim_calc.reset()
                ms_ssim_calc.reset()
                ssim_value = ssim_calc(imgs_s, imgs_r).item()
                ms_ssim_value = ms_ssim_calc(imgs_s, imgs_r).item()
                batch_size = imgs_r.shape[0]
                ssim_scores.extend([ssim_value] * batch_size)
                ms_ssim_scores.extend([ms_ssim_value] * batch_size)

        return float(np.mean(ssim_scores)), float(np.mean(ms_ssim_scores))

    # --- F. 1-NEAREST NEIGHBOR (1-NN) FEATURE ACCURACY ---
    def compute_1nn_accuracy(
        self,
        real_loader: DataLoader,
        syn_loader: DataLoader,
        medical_weights: Optional[Path] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> float:
        """Run a leave-one-out 1-NN two-sample test in RadImageNet space."""
        print("\n[Evaluating 1-Nearest Neighbor Feature Accuracy]")
        if medical_weights is None:
            raise ValueError(
                "1-NN requires a RadImageNet ResNet-50 checkpoint. "
                "Pass it with --med_fid_weights."
            )

        real_loader = self._sample_loader(real_loader, max_samples, seed)
        syn_loader = self._sample_loader(syn_loader, max_samples, seed + 1)
        print(
            f"1-NN sample size: real={len(real_loader.dataset)}, "
            f"synthetic={len(syn_loader.dataset)}; seeds: real={seed}, synthetic={seed + 1}"
        )
        feature_extractor = self._load_radimagenet_backbone(medical_weights)

        def extract_features(loader: DataLoader) -> torch.Tensor:
            features = []
            with torch.no_grad():
                for batch in tqdm(loader, desc="Extracting RadImageNet Features"):
                    images = self._as_radimagenet_input(batch["image"].to(self.device))
                    embeddings = feature_extractor(images).flatten(1)
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    features.append(embeddings.cpu())
            return torch.cat(features, dim=0)

        real_features = extract_features(real_loader)
        syn_features = extract_features(syn_loader)
        X = torch.cat([real_features, syn_features], dim=0).numpy()
        y = np.concatenate([
            np.zeros(len(real_features), dtype=np.int64),
            np.ones(len(syn_features), dtype=np.int64),
        ])

        from sklearn.neighbors import NearestNeighbors

        # Request two neighbours because the closest point to each query is
        # the query itself. The second neighbour is the leave-one-out choice.
        nearest = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nearest.fit(X)
        _, indices = nearest.kneighbors(X)
        predictions = y[indices[:, 1]]
        accuracy = float(np.mean(predictions == y))
        return accuracy


# ------------------------------------------------------------------------------
# 3. CLI ENTRY POINT & SUBCOMMAND ROUTING
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Quality & Metric Evaluator for Medical MRI Datasets")
    
    # Dataset Paths
    parser.add_argument("--real_dir", type=str, required=True, help="Path to Real root dir ('train' or 'test')")
    parser.add_argument("--syn_dir", type=str, default=None, help="Path to Synthetic root dir")
    parser.add_argument("--modality", type=str, default="bravo", help="Subfolder modality name to evaluate ('bravo')")
    parser.add_argument("--target_size", type=int, nargs=2, default=(256, 256), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument(
        "--resize_mode",
        choices=("pad_crop", "resize"),
        default="pad_crop",
        help="Spatial preprocessing before metric calculation: preserve pixel scale with pad/crop, or resample to target_size.",
    )
    parser.add_argument("--exclusion_file", type=str, default="corrupted_files.txt", help="Path to corrupted list .txt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for metric loader")
    parser.add_argument(
        "--fid_samples",
        type=int,
        default=None,
        help="Maximum samples per dataset for FID; use the same cap for both datasets (for example 400).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducible FID/MMD subsampling (synthetic data uses base seed + 1).",
    )
    
    # Metric Selection Flags
    parser.add_argument("--all", action="store_true", help="Run all available metrics")
    parser.add_argument("--is_score", action="store_true", help="Compute Inception Score")
    parser.add_argument("--fid", action="store_true", help="Compute Standard FID (Inception-v3)")
    parser.add_argument("--med_fid", action="store_true", help="Compute Medical FID (RadImageNet ResNet-50)")
    parser.add_argument(
        "--med_fid_weights",
        type=Path,
        default=None,
        help="Path to an official RadImageNet ResNet-50 PyTorch checkpoint.",
    )
    parser.add_argument("--cmmd", action="store_true", help="Compute CMMD (CLIP-based MMD)")
    parser.add_argument(
        "--cmmd_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP vision model used for CMMD.",
    )
    parser.add_argument(
        "--cmmd_sigma",
        type=float,
        default=None,
        help="RBF bandwidth for CMMD; default estimates it from median cross-dataset distances.",
    )
    parser.add_argument(
        "--cmmd_block_size",
        type=int,
        default=256,
        help="Kernel block size for CMMD memory control.",
    )
    parser.add_argument(
        "--cmmd_samples",
        type=int,
        default=None,
        help="Maximum samples per dataset for CMMD; use an equal cap such as 2000.",
    )
    parser.add_argument("--rad_mmd", action="store_true", help="Compute RadImageNet feature-space MMD")
    parser.add_argument(
        "--rad_mmd_sigma",
        type=float,
        default=None,
        help="RBF bandwidth for RadImageNet-MMD; default estimates it from median distances.",
    )
    parser.add_argument(
        "--rad_mmd_samples",
        type=int,
        default=None,
        help="Maximum samples per dataset for RadImageNet-MMD; use an equal cap such as 2000.",
    )
    parser.add_argument(
        "--one_nn_samples",
        type=int,
        default=None,
        help="Maximum samples per dataset for leave-one-out 1-NN; use an equal cap such as 2000.",
    )
    parser.add_argument("--ssim", action="store_true", help="Compute SSIM and MS-SSIM")
    parser.add_argument(
        "--ssim_paired",
        action="store_true",
        help="Confirm that real and synthetic loaders contain aligned image pairs for SSIM/MS-SSIM.",
    )
    parser.add_argument("--one_nn", action="store_true", help="Compute 1-Nearest Neighbor Accuracy")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Metric Evaluation Engine on Device: {device}")

    # Load Exclusion Rules
    exclusion_set = load_exclusion_list(Path(args.exclusion_file))

    # Instantiate Datasets
    real_ds = NiftiSliceDataset(
        Path(args.real_dir),
        exclusion_set=set(),
        modality=args.modality,
        target_size=tuple(args.target_size),
        resize_mode=args.resize_mode,
    )
    real_loader = DataLoader(real_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    syn_loader = None
    if args.syn_dir:
        syn_ds = NiftiSliceDataset(
            Path(args.syn_dir),
            exclusion_set=exclusion_set,
            modality=args.modality,
            target_size=tuple(args.target_size),
            resize_mode=args.resize_mode,
        )
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
            std_fid = evaluator.compute_fid(
                real_loader,
                syn_loader,
                use_medical_backbone=False,
                max_samples=args.fid_samples,
                seed=args.seed,
            )
            print(f"  Standard FID (Inception-v3): {std_fid:.4f}")

        if args.all or args.med_fid:
            m_fid = evaluator.compute_fid(
                real_loader,
                syn_loader,
                use_medical_backbone=True,
                max_samples=args.fid_samples,
                seed=args.seed,
                medical_weights=args.med_fid_weights,
            )
            print(f"  Medical FID (RadImageNet ResNet-50): {m_fid:.4f}")

        if args.all or args.cmmd:
            cmmd_val = evaluator.compute_cmmd(
                real_loader,
                syn_loader,
                clip_model_name=args.cmmd_model,
                sigma=args.cmmd_sigma,
                kernel_batch_size=args.cmmd_block_size,
                max_samples=args.cmmd_samples,
                seed=args.seed,
            )
            print(f"  CMMD (CLIP-MMD):             {cmmd_val:.6f}")

        if args.all or args.rad_mmd:
            rad_mmd_value = evaluator.compute_radimagenet_mmd(
                real_loader,
                syn_loader,
                medical_weights=args.med_fid_weights,
                sigma=args.rad_mmd_sigma,
                max_samples=args.rad_mmd_samples,
                seed=args.seed,
            )
            print(f"  RadImageNet-MMD:             {rad_mmd_value:.6f}")

        if args.all or args.ssim:
            if not args.ssim_paired:
                print("  SSIM/MS-SSIM skipped: datasets are not confirmed as paired and aligned.")
            else:
                ssim_v, ms_ssim_v = evaluator.compute_ssim_metrics(
                    real_loader,
                    syn_loader,
                    paired=True,
                )
                print(f"  Mean SSIM:                   {ssim_v:.4f}")
                print(f"  Mean MS-SSIM:                {ms_ssim_v:.4f}")

        if args.all or args.one_nn:
            one_nn_acc = evaluator.compute_1nn_accuracy(
                real_loader,
                syn_loader,
                medical_weights=args.med_fid_weights,
                max_samples=args.one_nn_samples,
                seed=args.seed,
            )
            print(f"  1-NN Feature Accuracy:       {one_nn_acc * 100:.2f}% (Target: 50.0%)")

    print("==================================================\n")


if __name__ == "__main__":
    main()