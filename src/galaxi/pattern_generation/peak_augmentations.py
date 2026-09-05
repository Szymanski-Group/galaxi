import torch
from typing import Tuple


class PeakAugmentor:
    """
    Fully GPU-accelerated overlapping peak augmentor.
    """

    def __init__(
        self,
        augmentation_intensity_threshold: float = 0.25,
        augmentation_fraction: float = 0.3,
        overlapping_peak_intensity_range: Tuple[float, float] = (0.2, 2.0),
        overlapping_peak_shift_range: Tuple[float, float] = (0.05, 0.5),
        cluster_tolerance: float = 0.2
    ):
        self.augmentation_intensity_threshold = augmentation_intensity_threshold
        self.augmentation_fraction = augmentation_fraction
        self.overlapping_peak_intensity_range = overlapping_peak_intensity_range
        self.overlapping_peak_shift_range = overlapping_peak_shift_range
        self.cluster_tolerance = cluster_tolerance

    # MAIN ENTRY
    def augment_peak_list(self, x, y, hkls, d_hkls):

        B, N = x.shape
        device = x.device

        max_intensity = y.max(dim=1, keepdim=True).values
        threshold = self.augmentation_intensity_threshold * max_intensity

        aug_x = []
        aug_y = []
        aug_h = []
        aug_d = []
        augmentation_info_list = []

        for b in range(B):

            xb = x[b]
            yb = y[b]
            hb = hkls[b]
            db = d_hkls[b]

            clusters = self._cluster_peaks_gpu(
                xb, yb, threshold[b]
            )

            if clusters.numel() == 0:
                aug_x.append(xb)
                aug_y.append(yb)
                aug_h.append(hb)
                aug_d.append(db)
                augmentation_info_list.append([])
                continue

            n_clusters = clusters.shape[0]
            k = max(1, int(n_clusters * self.augmentation_fraction))
            k = min(k, n_clusters)

            perm = torch.randperm(n_clusters, device=device)[:k]
            selected = clusters[perm]

            valid_mask = selected >= 0
            idx = selected[valid_mask]

            if idx.numel() == 0:
                aug_x.append(xb)
                aug_y.append(yb)
                aug_h.append(hb)
                aug_d.append(db)
                augmentation_info_list.append([])
                continue

            # generate shifts
            low_s, high_s = self.overlapping_peak_shift_range
            shift_mag = torch.empty(k, device=device).uniform_(low_s, high_s)
            shift_sign = torch.randint(0, 2, (k,), device=device) * 2 - 1
            shifts = shift_mag * shift_sign

            # generate intensity factors
            low_i, high_i = self.overlapping_peak_intensity_range
            factors = torch.empty(k, device=device).uniform_(low_i, high_i)

            expanded_shift = shifts.unsqueeze(1).expand_as(selected)[valid_mask]
            expanded_factor = factors.unsqueeze(1).expand_as(selected)[valid_mask]

            x_new = xb[idx] + expanded_shift
            y_new = yb[idx] * expanded_factor
            h_new = hb[idx]
            d_new = db[idx]

            aug_x.append(torch.cat([xb, x_new], dim=0))
            aug_y.append(torch.cat([yb, y_new], dim=0))
            aug_h.append(torch.cat([hb, h_new], dim=0))
            aug_d.append(torch.cat([db, d_new], dim=0))

            # record mean positions
            means = []
            for i, c in enumerate(selected):
                valid = c >= 0
                if valid.any():
                    means.append(float((xb[c[valid]] + shifts[i]).mean()))

            augmentation_info_list.append(means)

        # ----------- padding -----------
        max_len = max(v.shape[0] for v in aug_x)

        def pad(v, pad_value=0.0):
            if v.ndim == 1:
                P = max_len - v.shape[0]
                return torch.nn.functional.pad(v, (0, P), value=pad_value)
            else:
                P = max_len - v.shape[0]
                return torch.nn.functional.pad(v, (0, 0, 0, P), value=pad_value)

        X = torch.stack([pad(v) for v in aug_x], dim=0)
        Y = torch.stack([pad(v) for v in aug_y], dim=0)
        H = torch.stack([pad(v) for v in aug_h], dim=0)
        D = torch.stack([pad(v) for v in aug_d], dim=0)

        return X, Y, H, D, augmentation_info_list

    # GPU CLUSTERING
    def _cluster_peaks_gpu(self, x, y, threshold):

        device = x.device

        eligible = torch.where(y >= threshold)[0]
        if eligible.numel() == 0:
            return torch.empty((0, 0), dtype=torch.long, device=device)

        xs = x[eligible]
        order = torch.argsort(xs)
        sorted_idx = eligible[order]

        diffs = torch.abs(xs[order][1:] - xs[order][:-1])
        breaks = torch.where(diffs > self.cluster_tolerance)[0] + 1

        split_points = torch.cat([
            torch.tensor([0], device=device),
            breaks,
            torch.tensor([len(sorted_idx)], device=device)
        ])

        clusters = []
        max_size = 0

        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            cluster = sorted_idx[start:end]
            clusters.append(cluster)
            max_size = max(max_size, cluster.numel())

        cluster_tensor = torch.full(
            (len(clusters), max_size),
            -1,
            dtype=torch.long,
            device=device
        )

        for i, c in enumerate(clusters):
            cluster_tensor[i, :c.numel()] = c

        return cluster_tensor
