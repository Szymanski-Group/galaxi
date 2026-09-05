import torch
from typing import Dict, Any, List, Tuple


class PeakPerturbator:
    """
    Fully GPU-accelerated peak perturbator.
    """

    def __init__(
        self,
        perturbation_intensity_threshold: float = 0.2,
        perturbation_fraction_range: Tuple[float, float] = (0.0, 0.8),
        removal_probability: float = 0.5,
        shift_probability: float = 0.8,
        intensity_probability: float = 0.7,
        shift_range: Tuple[float, float] = (0.3, 1.5),
        intensity_factor_range: Tuple[float, float] = (2.0, 10.0)
    ):
        self.perturbation_intensity_threshold = perturbation_intensity_threshold
        self.perturbation_fraction_range = perturbation_fraction_range
        self.removal_probability = removal_probability
        self.shift_probability = shift_probability
        self.intensity_probability = intensity_probability
        self.shift_range = shift_range
        self.intensity_factor_range = intensity_factor_range


    # GPU CLUSTER CACHING
    def cache_peak_clusters(self, x: torch.Tensor, y: torch.Tensor, cluster_tolerance: float = 1.5):

        device = x.device
        max_intensity = y.max()
        threshold = self.perturbation_intensity_threshold * max_intensity

        eligible_mask = y >= threshold
        eligible_idx = torch.where(eligible_mask)[0]

        if eligible_idx.numel() == 0:
            return

        eligible_x = x[eligible_idx]
        sorted_order = torch.argsort(eligible_x)
        sorted_idx = eligible_idx[sorted_order]

        # Find cluster breaks
        diffs = torch.abs(x[sorted_idx][1:] - x[sorted_idx][:-1])
        breaks = torch.where(diffs > cluster_tolerance)[0] + 1

        split_points = torch.cat([
            torch.tensor([0], device=device),
            breaks,
            torch.tensor([len(sorted_idx)], device=device)
        ])

        clusters = []
        max_cluster_size = 0

        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            cluster = sorted_idx[start:end]
            clusters.append(cluster)
            max_cluster_size = max(max_cluster_size, cluster.numel())

        # Pad clusters to rectangular tensor
        num_clusters = len(clusters)
        cluster_tensor = torch.full(
            (num_clusters, max_cluster_size),
            -1,
            dtype=torch.long,
            device=device
        )

        cluster_sizes = torch.zeros(num_clusters, dtype=torch.long, device=device)

        for i, c in enumerate(clusters):
            cluster_tensor[i, :c.numel()] = c
            cluster_sizes[i] = c.numel()

        self.cluster_tensor = cluster_tensor
        self.cluster_sizes = cluster_sizes
        self.num_clusters = num_clusters

    # MAIN ENTRY
    def perturb_peak_list(
        self,
        x: torch.Tensor,   # (B,N)
        y: torch.Tensor,   # (B,N)
    ):

        self.cache_peak_clusters(x[0], y[0])

        device = x.device
        B, N = x.shape

        probs = torch.tensor(
            [self.removal_probability,
             self.shift_probability,
             self.intensity_probability],
            device=device
        )
        ptype = torch.multinomial(probs / probs.sum(), 1).item()

        if ptype == 0:
            self._apply_removal_gpu(y)

        elif ptype == 1:
            self._apply_shift_gpu(x)

        else:
            self._apply_intensity_gpu(y)

        # lightweight info only
        info = [{"type": ["removal", "shift", "intensity"][ptype]} for _ in range(B)]
        return x, y, info

    # CLUSTER SAMPLING
    def _sample_cluster_mask(self):

        device = self.cluster_tensor.device

        frac = torch.empty(1, device=device).uniform_(
            *self.perturbation_fraction_range
        )

        k = torch.clamp(
            (self.num_clusters * frac).long(),
            min=1
        )

        perm = torch.randperm(self.num_clusters, device=device)
        chosen = perm[:k]

        mask = torch.zeros(self.num_clusters, dtype=torch.bool, device=device)
        mask[chosen] = True
        return mask

    # REMOVAL
    def _apply_removal_gpu(self, y):

        B, N = y.shape
        device = y.device

        for b in range(B):

            cluster_mask = self._sample_cluster_mask()
            selected = self.cluster_tensor[cluster_mask]

            valid = selected >= 0
            idx = selected[valid]

            if idx.numel() > 0:
                y[b, idx] = 0.0

    # SHIFT
    def _apply_shift_gpu(self, x):

        B, N = x.shape
        device = x.device
        low, high = self.shift_range

        for b in range(B):

            cluster_mask = self._sample_cluster_mask()
            selected = self.cluster_tensor[cluster_mask]

            if selected.numel() == 0:
                continue

            valid = selected >= 0
            idx = selected[valid]

            shifts = torch.empty(
                selected.shape[0], device=device
            ).uniform_(low, high)

            signs = torch.randint(
                0, 2, (selected.shape[0],), device=device
            ) * 2 - 1

            shifts = shifts * signs

            expanded = shifts.unsqueeze(1).expand_as(selected)
            expanded = expanded[valid]

            x[b, idx] += expanded

    # INTENSITY
    def _apply_intensity_gpu(self, y):

        B, N = y.shape
        device = y.device
        low, high = self.intensity_factor_range

        for b in range(B):

            cluster_mask = self._sample_cluster_mask()
            selected = self.cluster_tensor[cluster_mask]

            if selected.numel() == 0:
                continue

            valid = selected >= 0
            idx = selected[valid]

            mag = torch.empty(
                selected.shape[0], device=device
            ).uniform_(low, high)

            invert = torch.rand_like(mag) < 0.5
            factors = torch.where(invert, 1.0 / mag, mag)

            expanded = factors.unsqueeze(1).expand_as(selected)
            expanded = expanded[valid]

            y[b, idx] *= expanded