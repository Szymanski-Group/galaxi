"""
Phase Detection Model for binary classification of XRD patterns.
"""

import copy
import os
import json
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam, AdamW, RMSprop, Adagrad, SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, log_loss
from typing import Dict, List, Optional, Any
from pathlib import Path
import zarr
import numpy as np
import torch

from ..core.model_base import BaseModel
from ..core.config import ModelConfig, DEFAULT_MODEL_CONFIG
from ..core.structures import StructureManager



class SpatialAttentionMask(nn.Module):
    '''
    Spatial attention module with masking support.
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_gate = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x, mask):
        scores = self.spatial_gate(x)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=2)      # (B, 1, L)
        x_weighted = x * weights        # (B, C, L)

        return x_weighted

class DetectionCNN(nn.Module):
    """CNN architecture for phase detection."""

    def __init__(self,
                 input_size: int = 7001,
                 conv_channels: List[int] = [16, 16, 16],
                 conv_kernels: List[int] = [27, 15, 11],
                 dilation_size: List[int] = None,
                 pool_size: List[int] = [2, 2, 1],
                 fc_sizes: List[int] = [16],
                 dropout_rate: float = 0.0,
                 activation: str = "relu",
                 use_batch_norm: bool = True,
                 use_mask: bool = True):
        super(DetectionCNN, self).__init__()

        # Activation function
        activation_key = activation.lower()
        if activation_key == "relu":
            self.activation = nn.ReLU()
        elif activation_key == "gelu":
            self.activation = nn.GELU()
        elif activation_key == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_key == "elu":
            self.activation = nn.ELU()
        elif activation_key == "swish":
            self.activation = nn.SiLU()  # SiLU is Swish
        elif activation_key == "tanh":
            self.activation = nn.Tanh()
        elif activation_key == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                f"Unknown activation '{activation}'. Supported: "
                "relu, leaky_relu, elu, gelu, swish, tanh, sigmoid."
            )

        self.conv_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        self.input_channels = 1
        in_channels = self.input_channels

        self.sigmoid = nn.Sigmoid()

        if dilation_size is None:
            dilation_size = [1] * len(conv_channels)

        # zip() below silently truncates to the shortest array if these are
        # mismatched, silently dropping conv layers instead of failing on a
        # genuine architecture-config error.
        array_lengths = {
            "conv_channels": len(conv_channels),
            "conv_kernels": len(conv_kernels),
            "pool_size": len(pool_size),
            "dilation_size": len(dilation_size),
        }
        if len(set(array_lengths.values())) > 1:
            raise ValueError(
                f"conv_channels, conv_kernels, pool_size, and dilation_size must all "
                f"have the same length, got: {array_lengths}"
            )
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")

        for i, (out_channels, kernel_size, pooling_layer) in enumerate(zip(conv_channels, conv_kernels, pool_size)):
            # ================= Convolutional layer =================
            conv_block = nn.Sequential()

            # Calculate dilation and padding
            dilation = dilation_size[i]
            padding = (kernel_size - 1) * dilation // 2

            # Convolution
            conv_block.add_module(f'conv_{i}',
                                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, padding=padding))

            # Batch normalization
            if use_batch_norm:
                conv_block.add_module(f'bn_{i}', nn.BatchNorm1d(out_channels))

            # Activation
            conv_block.add_module(f'activation_{i}', self.activation)

            # Dropout
            conv_block.add_module(f'dropout_{i}', nn.Dropout(dropout_rate))

            self.conv_layers.append(conv_block)

            in_channels = out_channels

            # pooling layer
            self.pooling_layers.append(nn.MaxPool1d(pooling_layer))


        # Apply mask attention at the last conv layer
        self.mask_attention = SpatialAttentionMask(out_channels)

        # Calculate actual flattened size using dummy forward pass
        self.flattened_size = self._calculate_flattened_size(input_size)

        if len(fc_sizes) > 0:
            # Build hidden fully connected layers
            self.fc_layers = nn.ModuleList()
            fc_input_size = self.flattened_size

            for i, fc_size in enumerate(fc_sizes):
                fc_block = nn.Sequential(
                    nn.Linear(fc_input_size, fc_size),
                    nn.BatchNorm1d(fc_size) if use_batch_norm else nn.Identity(),
                    self.activation,
                    nn.Dropout(dropout_rate),
                )
                self.fc_layers.append(fc_block)
                fc_input_size = fc_size

            self.output_layer = nn.Linear(fc_sizes[-1], 1)

        else:
            # No hidden fully connected layers — output directly from flatten
            self.fc_layers = nn.ModuleList()
            self.output_layer = nn.Linear(self.flattened_size, 1)

    def _calculate_flattened_size(self, input_size):
        """Calculate the actual flattened size mathematically without a dummy pass."""
        L = input_size
        C = self.input_channels

        # Iterate through layers to calculate shrinking spatial dimensions
        for i, conv_block in enumerate(self.conv_layers):
            # 1. Conv1d length calculation (First module in the Sequential block is always the primary Conv1d)
            conv_layer = conv_block[0]

            kernel_size = conv_layer.kernel_size[0]
            dilation = conv_layer.dilation[0]
            padding = conv_layer.padding[0]
            stride = conv_layer.stride[0]
            C = conv_layer.out_channels

            L = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

            # 2. MaxPool1d length calculation
            pool_layer = self.pooling_layers[i]

            pool_kernel = pool_layer.kernel_size
            pool_stride = pool_layer.stride
            pool_padding = pool_layer.padding
            pool_dilation = getattr(pool_layer, 'dilation', 1)

            # Handle PyTorch returning either integers or tuples for 1D pooling
            if isinstance(pool_kernel, tuple): pool_kernel = pool_kernel[0]
            if isinstance(pool_stride, tuple): pool_stride = pool_stride[0]
            if isinstance(pool_padding, tuple): pool_padding = pool_padding[0]
            if isinstance(pool_dilation, tuple): pool_dilation = pool_dilation[0]

            L = (L + 2 * pool_padding - pool_dilation * (pool_kernel - 1) - 1) // pool_stride + 1

        return C * L

    def forward(self, x, return_logit=False):

        # Ensure input has correct shape: (B, 1, N) or (B, 2, N) if masked
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Extract mask if present
        mask = None
        if x.size(1) == 2:
            mask = x[:, 1:, :]   # (B, 1, N)
            x = x[:, :1, :]    # (B, 1, N)

        # Convolutional layers
        for i, conv_layer in enumerate(self.conv_layers):
            # pass through conv_block
            conv_block = conv_layer
            for layer in conv_block:
                x = layer(x)

            # pooling layer
            pooling_layer = self.pooling_layers[i]
            x = pooling_layer(x)
            if mask is not None:
                mask = F.max_pool1d(mask,
                        kernel_size=pooling_layer.kernel_size,
                        stride=pooling_layer.stride,
                        padding=pooling_layer.padding)

                # Keep the mask length locked to the signal length. Conv1d is
                # built with padding=(k-1)//2, which preserves length for odd
                # kernels but loses one sample per layer for even ones, while
                # the mask path has no conv to shrink it. Re-aligning here lets
                # SpatialAttentionMask combine them for any kernel size; with
                # odd kernels the lengths already agree and this is a no-op.
                if mask.size(-1) != x.size(-1):
                    if mask.size(-1) > x.size(-1):
                        mask = mask[..., :x.size(-1)]
                    else:
                        mask = F.pad(mask, (0, x.size(-1) - mask.size(-1)), mode="replicate")

            # Apply mask attention at the last conv layer
            if i == len(self.conv_layers) - 1 and mask is not None:
                x = self.mask_attention(x, mask)

        # flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        # Output layer
        x = self.output_layer(x)

        if return_logit:
            return x.squeeze(1)
        else:
            x = self.sigmoid(x)
            return x.squeeze(1)

class PatternsDataset(torch.utils.data.Dataset):
    def __init__(self, paths=None, labels=None, num_points=None,
                 indices=None, patterns=None):
        """
        Two modes:
        - Disk mode: pass paths + labels
        - Cache mode: pass patterns + labels
        """
        self.num_points = num_points

        # ========== CACHE MODE ==========
        if patterns is not None:
            self.use_cache = True
            patterns = np.asarray(patterns, dtype=np.float32)
            labels = np.asarray(labels)

            if indices is not None:
                self.patterns = patterns[indices]
                self.labels   = labels[indices]
            else:
                self.patterns = patterns
                self.labels   = labels
            return

        # ========== DISK MODE ==========
        self.use_cache = False

        paths = list(paths)
        labels = np.asarray(labels)

        if indices is None:
            self.paths  = paths
            self.labels = labels
        else:
            self.paths  = [paths[i] for i in indices]
            self.labels = labels[indices]

        self._zarr_cache = {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]

        # ======= CACHE MODE =======
        if self.use_cache:
            y = self.patterns[idx]
            return torch.tensor(y, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

        # ======= DISK MODE =======
        p = self.paths[idx]

        # Zarr
        if isinstance(p, tuple):
            zarr_path, row_idx = p
            root = self._load_zarr(zarr_path)
            y = root["patterns"][row_idx]

        else:
            # xy or npy
            p = Path(p)
            if p.suffix == ".xy":
                y = np.loadtxt(p)
                # A genuine (2θ, intensity) pattern -- the conventional
                # meaning of ".xy", e.g. UnifiedPatternGenerator's own public
                # output -- looks nothing like this class's expected 2-row
                # (intensity, mask) cache format: one "row" is a monotonic
                # angle axis spanning several degrees, the other is a
                # near-binary 0/1 mask or a bounded intensity trace. Catch the
                # former here with a clear error instead of silently training
                # on (angle-as-intensity, intensity-as-mask). This pipeline's
                # own writers use .npy for the (intensity, mask) cache format
                # specifically to avoid this collision; a real .xy file
                # reaching here means it needs to go through
                # galaxi.core.pattern_utils.regularize_input first, not be
                # loaded as pre-cached training data directly.
                if y.ndim == 2 and y.shape[1] == 2:
                    angle_col = y[:, 0]
                    if angle_col.size > 1 and np.all(np.diff(angle_col) >= 0) and (angle_col.max() - angle_col.min()) > 5.0:
                        raise RuntimeError(
                            f"{p} looks like a raw (2θ, intensity) pattern (monotonic first "
                            f"column spanning {angle_col.max() - angle_col.min():.1f} degrees), not "
                            f"this dataset's expected preprocessed (intensity, mask) cache format. "
                            f"Run it through galaxi.core.pattern_utils.regularize_input first, or "
                            f"regenerate training data with file_format='npy'."
                        )
            elif p.suffix == ".npy":
                y = np.load(p)
            else:
                raise RuntimeError(f"Unsupported format: {p}")
        # transpose
        y = y.T     # (1, N) or (2, N)

        # shape check
        expected_len = int(self.num_points)

        # Check if the expected length exists in the array's shape
        if expected_len not in y.shape:
            raise RuntimeError(f"Pattern length mismatch: expected {expected_len}, got shape {y.shape}")

        if y.ndim not in [1, 2]:
            raise RuntimeError(f"Pattern dimension mismatch: expected 1D or 2D, got {y.ndim}D")

        if y.ndim == 2 and y.shape[0] == expected_len:
            y = y.T  # Transpose the array

        return torch.tensor(y, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class PhaseDetectionModel(BaseModel):
    """Phase detection model for binary classification."""

    def __init__(self,
                 target_phase: str,
                 reference_dir: Optional[str] = None,
                 config: Optional[ModelConfig] = None,
                 use_gpu=True):

        self.model_dtype = torch.float32
        # Deep-copy when falling back to the shared DEFAULT_MODEL_CONFIG singleton:
        # load_model() mutates self.config's fields in place (input_size, min_angle,
        # etc.), so aliasing the global default here would let one model's loaded
        # config silently leak into every other instance that didn't pass its own.
        self.config = copy.deepcopy(config) if config is not None else copy.deepcopy(DEFAULT_MODEL_CONFIG)
        self.target_phase = target_phase
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model = None
        self.training_history = None

        # Only initialize structure manager if reference_dir is provided
        if reference_dir is not None:
            self.structure_manager = StructureManager(reference_dir)
        else:
            self.structure_manager = None

        # Caching options. Caching to speed up data generation if enabled
        self.cache_training_data = None
        self.cached_patterns = {"positive": [], "negative": []}
        self.cached_labels = []

    def _load_pattern_paths(self, generation_output_dir):
        """
        Collects paths to individual patterns for XY, NPY, Zarrformats.

        Zarr files produce tuples: (zarr_path, row_index)
        """
        paths = []
        labels = []

        for subfolder, label in (("positive", 1), ("negative", 0)):
            pdir = Path(generation_output_dir) / subfolder

            # xy
            for p in pdir.glob("*.xy"):
                paths.append(p)
                labels.append(label)

            # npy
            for p in pdir.glob("*.npy"):
                paths.append(p)
                labels.append(label)

            # zarr
            zarr_path = pdir / "patterns.zarr"
            if zarr_path.exists():
                g = zarr.open_group(str(zarr_path), mode="r")
                if "patterns" not in g:
                    raise RuntimeError(f"Zarr file {zarr_path} missing 'patterns' array")

                n_rows = g["patterns"].shape[0]
                for i in range(n_rows):
                    paths.append((str(zarr_path), i))  # (path_to_zarr, index)
                    labels.append(label)

        return paths, np.array(labels)


    def train(self, **kwargs) -> Dict[str, Any]:
        """
        Abstract method implementation - delegates to train_detection.

        This method exists to satisfy the BaseModel abstract interface and
        accepts exactly what train_detection() accepts. Call train_detection()
        directly for the actual detection model training.
        """
        return self.train_detection(**kwargs)

    def train_detection(self,
                       references_dir: str = None,
                       generation_output_dir: Path = None,
                       model_output_dir: str = "detection_models",
                       test_size: float = 0.2,
                       val_size: float = 0.2,
                       workflow_config: Optional[Dict[str, Any]] = None,
                       use_external_test_set: bool = False,
                       external_generation_output_dir: Path = None,
                       ) -> Dict[str, Any]:
        """
        Train phase detection model.

        Args:
            references_dir: Path to reference data
            generation_output_dir: Path to generated training patterns
            model_output_dir: Directory to save models
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            workflow_config: complete workflow_config for reference
            use_external_test_set: If True, uses external test patterns/labels
            external_generation_output_dir: Path to external generation data

        Returns:
            Training results dictionary
        """

        # Combine positive and negative patterns
        pattern_paths, labels = self._load_pattern_paths(generation_output_dir)

        if use_external_test_set:
            ext_paths, ext_labels = self._load_pattern_paths(external_generation_output_dir)
            if ext_paths is None or ext_labels is None:
                raise ValueError("External test data required when use_external_test_set=True")

        # Build cached_data
        cached_data = None
        cached_labels = None

        if self.cache_training_data:
            pos = self.cached_patterns["positive"]
            neg = self.cached_patterns["negative"]

            cached_data = np.asarray(pos + neg, dtype=np.float32)

            # Build labels that match cached_data
            cached_labels = np.array(
                [1] * len(pos) + [0] * len(neg),
                dtype=np.int64
            )

        # Main dataset size
        n_total = len(cached_data) if cached_data is not None else len(pattern_paths)
        # Splits
        n_val = int(n_total * val_size)
        n_test = int(n_total * test_size) if not use_external_test_set else 0
        indices = np.random.permutation(n_total)

        train_idx = indices[n_val + n_test:]
        val_idx   = indices[:n_val]

        # Held-out test indices: drawn from the external set when one is
        # supplied, otherwise carved out of the same permutation as train/val.
        if use_external_test_set:
            test_idx = np.random.permutation(len(ext_paths))
        else:
            test_idx = indices[n_val:n_val + n_test]

        def make_dataset(paths, labels, idx):
            if cached_data is not None:
                return PatternsDataset(
                    patterns=cached_data,
                    labels=cached_labels,
                    num_points=self.config.input_size,
                    indices=idx
                )
            else:
                return PatternsDataset(
                    paths=paths,
                    labels=labels,
                    num_points=self.config.input_size,
                    indices=idx
                )

        # Build datasets
        train_dataset = make_dataset(pattern_paths, labels, train_idx)
        val_dataset   = make_dataset(pattern_paths, labels, val_idx)

        if use_external_test_set:
            test_dataset = PatternsDataset(
                paths=ext_paths,
                labels=ext_labels,
                num_points=self.config.input_size,
                indices=test_idx
            )
        else:
            test_dataset = make_dataset(pattern_paths, labels, test_idx)

        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")

        # Guard against degenerate splits: an empty split breaks the
        # training/eval loops below (division by zero, empty DataLoader),
        # and a single-class val/test split makes roc_auc_score raise and
        # produces meaningless precision/recall (both 0/0, silently masked
        # by the +1e-9 epsilon in the metrics below).
        train_labels_arr = np.asarray(cached_labels if cached_data is not None else labels)
        test_labels_arr = np.asarray(ext_labels) if use_external_test_set else train_labels_arr

        splits_to_check = [
            ("Training", train_idx, train_labels_arr),
            ("Validation", val_idx, train_labels_arr),
            ("Test", test_idx, test_labels_arr),
        ]
        for split_name, idx, labels_arr in splits_to_check:
            if len(idx) == 0:
                raise ValueError(
                    f"{split_name} split is empty ({n_total} total samples, "
                    f"val_size={val_size}, test_size={test_size}) -- cannot train/evaluate."
                )
            if split_name != "Training":
                split_labels = labels_arr[idx]
                if len(np.unique(split_labels)) < 2:
                    raise ValueError(
                        f"{split_name} split contains only class {set(split_labels.tolist())} "
                        f"({len(idx)} samples) -- need both positive and negative examples to "
                        f"compute meaningful accuracy/AUC/precision/recall. Increase the dataset "
                        f"size or split fraction for this phase."
                    )

        # Create model
        self.model = DetectionCNN(
            input_size=self.config.input_size,
            conv_channels=self.config.conv_channels,
            conv_kernels=self.config.conv_kernels,
            dilation_size=self.config.dilation_size,
            pool_size=self.config.pool_size,
            fc_sizes=self.config.fc_sizes,
            dropout_rate=self.config.dropout_rate,
            activation=self.config.activation,
            use_batch_norm=self.config.use_batch_norm,
            use_mask=self.config.use_mask
        ).to(self.device)

        self.model_dtype = next(self.model.parameters()).dtype

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = self.build_optimizer(self.model, self.config)

        # Training loop
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        val_precisions=[]        # precision=TP/(TP+FP)
        val_recalls=[]           # recall=TP/(TP+FN)

        batch_size = self.config.batch_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        # Defining early stopping parameters
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        # Defining learning rate scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            threshold=1e-4,
            cooldown=0,
            min_lr=1e-6,
        )

        # Training loop
        print("Starting training...")
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0
            epoch_train_loss = 0
            epoch_train_correct = 0
            epoch_train_total = 0
            num_batches = len(train_loader)

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(dtype=self.model_dtype, device=self.device)
                batch_y = batch_y.to(dtype=torch.float32, device=self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X).view(-1)  # (b, )
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    predictions = (outputs > 0.5).float()
                    epoch_train_correct += (predictions == batch_y).sum().item()
                    epoch_train_total += batch_y.size(0)

                epoch_train_loss += loss.item()

            if num_batches > 0:
                avg_train_loss = epoch_train_loss / num_batches
            else:
                avg_train_loss = float('nan')

            train_losses.append(avg_train_loss)
            train_accuracy = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0
            train_accuracies.append(train_accuracy)

            # Validation
            self.model.eval()
            with torch.no_grad():
                TP, FP, FN = 0, 0, 0
                correct, total = 0, 0
                val_epoch_loss = 0
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(dtype=self.model_dtype, device=self.device)
                    batch_y = batch_y.to(dtype=torch.float32, device=self.device) # BCELoss expects float32 targets
                    val_outputs = self.model(batch_X).view(-1)      # (b, )
                    val_loss = criterion(val_outputs, batch_y)
                    val_epoch_loss += val_loss.item()
                    val_predictions = (val_outputs > 0.5).float()
                    correct += (val_predictions == batch_y).sum().item()
                    total += batch_y.size(0)
                    TP += ((val_predictions == 1) & (batch_y == 1)).sum().item()
                    FP += ((val_predictions == 1) & (batch_y == 0)).sum().item()
                    FN += ((val_predictions == 0) & (batch_y == 1)).sum().item()

                avg_val_loss = val_epoch_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                val_accuracy = correct / total
                val_accuracies.append(float(val_accuracy))

                # Precision and recall
                val_precision = TP / (TP + FP + 1e-9)
                val_recall = TP / (TP + FN + 1e-9)

                # Store for logging
                val_precisions.append(val_precision)
                val_recalls.append(val_recall)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # state_dict() returns references to the live parameter tensors,
                # not copies -- optimizer.step() updates those same tensors
                # in place on later epochs, silently corrupting an uncloned
                # "checkpoint". Detach + clone each tensor so this snapshot is
                # independent of subsequent training.
                best_model_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

                if self.config.early_stopping_patience is not None:
                    if epochs_no_improve >= self.config.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}/{self.config.num_epochs}")
                        break

            # Step the scheduler
            scheduler.step(avg_val_loss)  # monitor val_loss
            current_lr = optimizer.param_groups[0]['lr']

            # Print epoch results on separate line
            print(f"Epoch {epoch+1:2d}/{self.config.num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}")

        # Restore the best validation checkpoint before evaluating/saving --
        # otherwise test metrics and the saved model reflect the final (or
        # early-stopped) epoch's weights instead of the best one tracked above.
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Test set evaluation
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(dtype=self.model_dtype, device=self.device)
                y = y.to(dtype=torch.float32, device=self.device)

                outputs = self.model(x).view(-1)    # raw predictions
                all_preds.append(outputs.cpu())
                all_labels.append(y.cpu())

        test_outputs_np = torch.cat(all_preds).numpy()
        test_labels = torch.cat(all_labels).numpy()

        test_predictions = (test_outputs_np > 0.5).astype(np.float32)
        test_accuracy = (test_predictions == test_labels).astype(np.float32).mean()
        test_auc = roc_auc_score(test_labels, test_outputs_np)
        test_logloss = log_loss(test_labels, test_outputs_np)

        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test AUC: {test_auc:.4f}")
        print(f"Final Test Log Loss: {test_logloss:.4f}")

        # Save results
        os.makedirs(model_output_dir, exist_ok=True)

        # Save best model if save_models is True
        if self.config.save_models and best_model_state is not None:
            model_path = os.path.join(model_output_dir, f"detection_model_{self.target_phase}.pth")
            torch.save(best_model_state, model_path)

        # Save model configuration
        config_path = os.path.join(model_output_dir, f"detection_model_{self.target_phase}_config.json")
        model_config = {
            'smoothing_window_length': self.config.smoothing_window_length,
            "snip_iter": self.config.snip_iter,
            "noise_sensitivity": self.config.noise_sensitivity,
            "gate_sharpness": self.config.gate_sharpness,
            "magnification_power": self.config.magnification_power,
            'conv_channels': self.config.conv_channels,
            'conv_kernels': self.config.conv_kernels,
            'dilation_size': self.config.dilation_size,
            'pool_size': self.config.pool_size,
            'fc_sizes': self.config.fc_sizes,
            'activation': self.config.activation,
            'use_batch_norm': self.config.use_batch_norm,
            'dropout_rate': self.config.dropout_rate,
            'optimizer': self.config.optimizer,
            'num_epochs': self.config.num_epochs,
            'learning_rate': self.config.learning_rate,
            'batch_size': batch_size,
            'early_stopping_patience': self.config.early_stopping_patience,
            'input_size': self.config.input_size,
            'min_angle': self.config.min_angle,
            'max_angle': self.config.max_angle,
            'target_phase': self.target_phase,
            'use_mask': self.config.use_mask
        }
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)

        # Save training history
        history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_precisions': val_precisions,
            'val_recalls': val_recalls
        }

        history_path = os.path.join(model_output_dir, f"training_history_{self.target_phase}.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Save the whole workflow_config to ensure repeatability
        if workflow_config is not None:
            workflow_config_path = os.path.join(model_output_dir, f"workflow_config_{self.target_phase}.json")
            with open(workflow_config_path, 'w') as f:
                json.dump(workflow_config, f, indent=2)

        # Generate classification report
        report = classification_report(test_labels, test_predictions, output_dict=True)

        final_report = {
            'target_phase': self.target_phase,
            'train': {
                'train_loss': train_losses[-1] if train_losses else 0.0,
                'train_accuracy': train_accuracies[-1] if train_accuracies else 0.0,
            },
            'validation': {
                'val_loss': val_losses[-1] if val_losses else 0.0,
                'val_accuracy': val_accuracies[-1] if val_accuracies else 0.0,
                'val_precision': val_precisions[-1] if val_precisions else 0.0,
                'val_recall': val_recalls[-1] if val_recalls else 0.0,
            },
            'test': {
                'test_accuracy': test_accuracy.item(),
                'test_auc': test_auc,
                'test_log_loss': test_logloss,
                **report
            }
        }

        report_path = os.path.join(model_output_dir, f"classification_report_{self.target_phase}.json")
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)

        self.save_peak_list(model_output_dir)

        # Save plots if requested
        if self.config.save_plots:
            self._save_training_plots(history, model_output_dir)
            self._save_roc_curve(test_labels, test_outputs_np, model_output_dir)

        return final_report

    def save_peak_list(self, output_dir: str):
        """
        Saves the target phase's 2θ and intensity lists to peak_list.json
        by simulating the XRD pattern using pymatgen's XRDCalculator.
        """
        theta_list, intensity_list = [], []

        try:
            # Inline imports to ensure safety if pymatgen is environment-specific
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            from pymatgen.core import Structure

            if self.structure_manager:
                # Retrieve the structure representation from your manager
                structure_or_path = self.structure_manager.get_structure(self.target_phase)

                # Robust handling: check if it's a file path string/Path object, or a Structure instance
                if isinstance(structure_or_path, (str, Path)):
                    structure = Structure.from_file(str(structure_or_path))
                else:
                    structure = structure_or_path

                # Initialize the calculator using standard CuKa radiation
                xrd_calc = XRDCalculator(wavelength="CuKa")
                pattern = xrd_calc.get_pattern(structure)

                # Extract and convert arrays to standard python lists for JSON serialization
                theta_list = pattern.x.tolist()
                intensity_list = pattern.y.tolist()
            else:
                print(f"Warning: StructureManager not initialized. Cannot generate peak list for {self.target_phase}.")

        except Exception as e:
            print(f"Warning: Could not simulate XRD pattern for {self.target_phase}: {e}")

        output_path = os.path.join(output_dir, "peak_list.json")
        with open(output_path, 'w') as f:
            json.dump({
                "phase_name": self.target_phase,
                "theta": theta_list,
                "intensity": intensity_list
            }, f, indent=4)

    def build_optimizer(self, model, config):
        lr = config.learning_rate

        if config.optimizer == "adam":
            opt = Adam(model.parameters(), lr=lr)

        elif config.optimizer == "adamw":
            opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

        elif config.optimizer == "rmsprop":
            opt = RMSprop(model.parameters(), lr=lr, alpha=0.99)

        elif config.optimizer == "adagrad":
            opt = Adagrad(model.parameters(), lr=lr)

        elif config.optimizer == "sgd":
            opt = SGD(model.parameters(), lr=lr, momentum=0.9)

        elif config.optimizer == "adabelief":
            # simple import (pip install adabelief-pytorch)
            opt = AdaBelief(model.parameters(), lr=lr, weight_decay=1e-4, rectify=False)
        else:
            raise ValueError(f"Unknown optimizer {config.optimizer}")


        return opt

    def load_model(self, model_path: str):
        """Load a pre-trained detection model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Try to load model configuration from training history
        config_path = model_path.replace('.pth', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)

            # Update current config with saved values
            conv_channels = saved_config.get('conv_channels', self.config.conv_channels)
            conv_kernels = saved_config.get('conv_kernels', self.config.conv_kernels)
            dilation_size = saved_config.get('dilation_size', self.config.dilation_size)
            pool_size = saved_config.get('pool_size', self.config.pool_size)
            fc_sizes = saved_config.get('fc_sizes', self.config.fc_sizes)
            dropout_rate = saved_config.get('dropout_rate', self.config.dropout_rate)
            activation = saved_config.get('activation', self.config.activation)
            use_batch_norm = saved_config.get('use_batch_norm', self.config.use_batch_norm)
            use_mask = saved_config.get('use_mask', self.config.use_mask)
            input_size = saved_config.get('input_size', 4501)

            # min_angle/max_angle determine the 2θ grid regularize_input
            # resamples onto before this model ever sees a pattern -- getting
            # this wrong silently collapses every prediction toward zero
            # (looks exactly like a bad/untrained model, not a preprocessing
            # mismatch) since older saved configs never recorded these two
            # fields. Warn loudly rather than silently falling back to a
            # generic default that may not match how this model was trained.
            if 'min_angle' in saved_config and 'max_angle' in saved_config:
                min_angle = saved_config['min_angle']
                max_angle = saved_config['max_angle']
            else:
                min_angle = self.config.min_angle
                max_angle = self.config.max_angle
                print(
                    f"Warning: {config_path} has no saved min_angle/max_angle "
                    f"(older model export). Falling back to config defaults "
                    f"({min_angle}-{max_angle} deg) -- if this model's training "
                    f"catalog actually used a different angular range, "
                    f"predictions will be near-meaningless. Pass the correct "
                    f"range explicitly if known."
                )

            # Keep self.config in sync with what was actually loaded, so
            # downstream code (e.g. regularize_input(..., min_angle=
            # model.config.min_angle, ...)) reflects this model's real
            # architecture/preprocessing rather than whatever ModelConfig
            # happened to be passed at construction time.
            self.config.input_size = input_size
            self.config.min_angle = min_angle
            self.config.max_angle = max_angle
            self.config.use_mask = use_mask
        else:
            # Fallback when no sidecar config file exists. Every architecture
            # parameter DetectionCNN needs below must be defined on this branch,
            # so fall back to self.config's own values -- the same pattern as
            # the saved_config.get(..., self.config.X) calls used above.
            input_size = 4501
            min_angle = self.config.min_angle
            max_angle = self.config.max_angle
            conv_channels = self.config.conv_channels
            conv_kernels = self.config.conv_kernels
            dilation_size = self.config.dilation_size
            pool_size = self.config.pool_size
            fc_sizes = self.config.fc_sizes
            dropout_rate = self.config.dropout_rate
            activation = self.config.activation
            use_batch_norm = self.config.use_batch_norm
            use_mask = self.config.use_mask
            print("Warning: No model config found, using default parameters")

        # Create model with correct architecture
        self.model = DetectionCNN(
            input_size=input_size,
            conv_channels=conv_channels,
            conv_kernels=conv_kernels,
            dilation_size=dilation_size,
            pool_size=pool_size,
            fc_sizes=fc_sizes,
            dropout_rate=dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_mask=use_mask,
        ).to(self.device)

        # Load state dict with error handling
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
        except RuntimeError as e:
            # strict=False alone only tolerates missing/unexpected keys, not a
            # same-key tensor with a different shape (e.g. a different
            # conv_channels/fc_sizes architecture) -- load_state_dict still
            # raises RuntimeError for those regardless of strict, so drop any
            # shape-mismatched entries ourselves before the strict=False call
            # actually has something loadable to work with.
            print(f"Warning: Model architecture mismatch ({e}), attempting partial load...")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model_state = self.model.state_dict()
            shape_mismatched = [
                k for k in state_dict
                if k in model_state and state_dict[k].shape != model_state[k].shape
            ]
            for k in shape_mismatched:
                del state_dict[k]
            if shape_mismatched:
                print(f"Shape-mismatched keys (not loaded): {shape_mismatched}")

            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

            self.model.eval()
            print(f"Detection model partially loaded from {model_path}")
            print("Warning: Some layers may not have loaded correctly")

    def predict(self, pattern: np.ndarray) -> float:
        """
        Predict probability that pattern contains target phase.
        Args:
            pattern: XRD pattern intensities

        Returns:
            Probability of containing target phase (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        self.model.eval()
        with torch.no_grad():
            pattern_tensor = torch.tensor(pattern, dtype=self.model_dtype, device=self.device).unsqueeze(0)  # (1, 1, N) or (1, 2, N) depending on mask
            output = self.model(pattern_tensor)

            # Handle both single predictions and batch predictions
            output_np = output.cpu().numpy()
            if output_np.ndim == 0:
                return output_np.item()
            else:
                return output_np[0]

    def _save_training_plots(self, history: Dict, output_dir: str):
        """Save training loss plots."""
        plt.figure(figsize=(8, 6))

        # Loss plot
        plt.plot(history['train_losses'], label='Training Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch', labelpad=16)
        plt.ylabel('Loss', labelpad=16)
        plt.title(f'{self.target_phase}')
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.1)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_plots_{self.target_phase}.png'), dpi=300)
        plt.close()

    def _save_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, output_dir: str):
        """Save ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate', labelpad=16)
        plt.ylabel('True Positive Rate', labelpad=16)
        plt.title(f'{self.target_phase}')
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_curve_{self.target_phase}.png'), dpi=300)
        plt.close()
