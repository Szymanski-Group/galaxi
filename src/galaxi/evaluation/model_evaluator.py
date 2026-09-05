"""
Comprehensive model evaluation framework for phase detection models.
"""

from __future__ import annotations

import os
import json
import tempfile
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, fields as dataclass_fields
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.graph_objects as go
import torch
from dara import search_phases
from galaxi.detection.detection_model import PhaseDetectionModel

from ..core.config import ModelConfig, XRDGenerationConfig, DaraConfig
from ..core.pattern_utils import build_phase_groups_from_peaks, preprocess_xrd_pattern, calculate_pattern_f1_metrics, generate_experimental_summary_csv, regularize_input, group_phases_func, count_header_lines


def _filter_to_model_config_fields(config: dict) -> dict:
    """Drop any keys a saved detection_model_*_config.json has that
    ModelConfig no longer defines (e.g. from a removed architecture option),
    so old on-disk model configs stay loadable after such a field is
    removed, rather than erroring on an unexpected keyword argument."""
    valid = {f.name for f in dataclass_fields(ModelConfig)}
    return {k: v for k, v in config.items() if k in valid}


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    confusion_matrix: np.ndarray
    classification_report: str

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'accuracy': float(self.accuracy),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1_score),
            'auc_score': float(self.auc_score),
            'confusion_matrix': self.confusion_matrix.tolist(),
            'classification_report': self.classification_report
        }

class ModelEvaluator:
    """Comprehensive evaluation framework for phase detection models."""

    def __init__(self,
                 phases: list = None,
                 ref_dir: str = None,
                 models_dir: str = ".",
                 output_dir: str = "evaluation_results",
                 xrd_config: Optional['XRDGenerationConfig'] = None,
                 model_config: ModelConfig = None,
                 hyperparameter_dict: Optional[Dict[str, Union[float, int]]] = None,
                 use_ensemble: bool = False):
        """
        Initialize the model evaluator.

        Args:
            models_dir: Directory containing trained models
            output_dir: Directory to save evaluation results
            xrd_config: XRD generation configuration (if None, uses default values)
            hyperparameter_dict: Serves to search models under proper hyperparameter combination
            use_ensemble: Use ensemble predictions if use_ensemble=True
        """
        self.phases = phases
        self.ref_dir = Path(ref_dir) if ref_dir else None
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.hyperparameter_dict = hyperparameter_dict
        self.use_ensemble = use_ensemble

        # Store XRD config for consistent angular range
        if xrd_config is not None:
            self.xrd_config = xrd_config
        else:
            # Import here to avoid circular imports
            from ..core.config import XRDGenerationConfig
            self.xrd_config = XRDGenerationConfig()  # Default config
            print(
                f"Warning: no xrd_config passed to ModelEvaluator -- defaulting to "
                f"{self.xrd_config.min_angle}-{self.xrd_config.max_angle} deg. "
                f"regularize_input resamples every pattern onto THIS range before "
                f"a model ever sees it, independent of each model's own saved "
                f"config -- if the models being evaluated were trained on a "
                f"different angular range (the pretrained catalog uses 5-105 deg), "
                f"every prediction will silently collapse toward zero. "
                f"Pass xrd_config=XRDGenerationConfig(min_angle=..., max_angle=...) "
                f"explicitly matching the models you're loading."
            )

        # model_config carries the preprocessing hyperparameters
        # (snip_iter, noise_sensitivity, gate_sharpness, magnification_power,
        # smoothing_window_length) that preprocess_xrd_pattern() needs. Resolve
        # it here, before any model is loaded, so a missing config is reported
        # up front rather than on the first pattern.
        self.model_config = model_config
        if self.model_config is None:
            from ..core.config import DEFAULT_MODEL_CONFIG
            self.model_config = DEFAULT_MODEL_CONFIG
            print(
                f"Warning: no model_config passed to ModelEvaluator -- defaulting to "
                f"DEFAULT_MODEL_CONFIG's preprocessing hyperparameters (snip_iter="
                f"{self.model_config.snip_iter}, noise_sensitivity="
                f"{self.model_config.noise_sensitivity}, gate_sharpness="
                f"{self.model_config.gate_sharpness}, magnification_power="
                f"{self.model_config.magnification_power}). If the models being "
                f"evaluated were trained with different preprocessing, every "
                f"prediction will be out-of-distribution. Pass model_config= "
                f"explicitly matching the models you're loading."
            )

        # Find available models
        self.available_models = self._find_available_models()

    def _find_available_models(self) -> Dict[str, Path]:
        """Find all available trained models."""
        models = {}
        print(f"Searching for trained models in {self.models_dir}")

        for phase_dir in self.models_dir.glob("models_*"):
            if not phase_dir.is_dir():
                continue

            phase_name = phase_dir.name.replace("models_", "")
            model_list = []

            # Determine HP subdirectory (if specified)
            hp_dir = phase_dir
            if self.hyperparameter_dict:
                hp_name = "_".join(f"{k}_{v}" for k, v in self.hyperparameter_dict.items())
                hp_dir = phase_dir / hp_name

            # Ensemble mode
            if self.use_ensemble:
                for ens_dir in hp_dir.glob("models_ensemble_*"):
                    model_file = ens_dir / f"detection_model_{phase_name}.pth"
                    if model_file.exists():
                        model_list.append(model_file)

                if model_list:
                    models[phase_name] = model_list

            # Single model mode
            else:
                model_file = hp_dir / f"detection_model_{phase_name}.pth"
                if model_file.exists():
                    models[phase_name] = model_file

        print(f"Found {len(models)} trained models")
        return models


    def load_patterns_from_directory(self, pattern_dir: str,
                                     file_extension: str = "xy") -> Tuple[List[np.ndarray], List[str]]:
        """
        Load patterns from a directory structure with proper 2θ range filtering.
        """
        patterns = []
        filenames = []

        pattern_path = Path(pattern_dir)

        # Recursively find all pattern files
        for pattern_file in pattern_path.rglob(f"*.{file_extension}"):
            try:
                data = np.loadtxt(pattern_file, skiprows=count_header_lines(pattern_file))
                assert data.ndim == 2, (
                    "Provided XRD patterns must have two columns: 2θ and intensity."
                )

                two_theta = data[:, 0]
                intensity = data[:, 1]
                pattern = np.stack((two_theta, intensity), axis=1)      # (N, 2)

                # Append a filename only alongside a pattern that was actually
                # kept, so the two lists stay the same length and every later
                # zip(patterns, filenames) pairing stays aligned.
                if np.any(intensity < 0):
                    print(f"Warning: Negative intensities in {pattern_file.name}")
                else:
                    patterns.append(pattern)
                    # Path relative to the search root, not the bare basename.
                    # rglob() descends into subdirectories and the comprehensive
                    # test sets are nested (positive/single_phase/strain/...);
                    # relative paths match how ground-truth files key their
                    # entries and stay unambiguous across sibling directories.
                    # For a flat directory this is identical to the basename.
                    filenames.append(pattern_file.relative_to(pattern_path).as_posix())

            except Exception as e:
                print(f"Warning: Could not load {pattern_file}: {e}")

        return patterns, filenames

    def load_ground_truth(self, ground_truth_file: str) -> Dict:
        """Load ground truth data from JSON file."""
        with open(ground_truth_file, 'r') as f:
            return json.load(f)

    def extract_labels_from_ground_truth(self, ground_truth: Dict, filenames: List[str], target_phase: str) -> List[int]:
        """
        Extract binary labels from ground truth for a specific target phase.

        Args:
            ground_truth: Ground truth dictionary
            filenames: List of pattern filenames
            target_phase: Target phase name

        Returns:
            List of binary labels (1 for target present, 0 for absent)
        """
        labels = []
        matched = 0

        # Ground-truth files key on the path relative to the test directory.
        # Index by basename as well, so a ground truth written with bare
        # filenames still resolves -- but only as a fallback, and only when the
        # basename is unambiguous.
        by_basename = {}
        for key in ground_truth:
            base = key.replace("\\", "/").rsplit("/", 1)[-1]
            by_basename.setdefault(base, []).append(key)

        for filename in filenames:
            # Normalize filename path
            norm_filename = filename.replace("\\", "/")

            gt_data = ground_truth.get(norm_filename)
            if gt_data is None:
                candidates = by_basename.get(norm_filename.rsplit("/", 1)[-1], [])
                if len(candidates) == 1:
                    gt_data = ground_truth[candidates[0]]

            if gt_data is not None:
                matched += 1

                # Check if target phase is present
                if 'target_present' in gt_data:
                    label = 1 if gt_data['target_present'] and gt_data.get('target_phase') == target_phase else 0
                else:
                    # Fallback: check if target phase name is in filename or metadata
                    label = 1 if target_phase in norm_filename else 0

                labels.append(label)
            else:
                # Fallback: infer from filename
                label = 1 if target_phase in norm_filename else 0
                labels.append(label)

        # A ground truth that matches nothing is a wiring error, not a dataset
        # of all-negatives. Labelling everything 0 would yield a plausible-looking
        # score over a single-class confusion matrix, so fail loudly instead.
        if ground_truth and filenames and matched == 0:
            example_gt = next(iter(ground_truth))
            raise ValueError(
                f"Ground truth matched none of the {len(filenames)} loaded patterns "
                f"for {target_phase!r}: no key corresponds to a pattern filename. "
                f"Ground-truth keys look like {example_gt!r}; pattern names look "
                f"like {filenames[0]!r}. These must use the same convention "
                "(path relative to the pattern directory)."
            )

        return labels


    def calculate_metrics(self, y_true: List[int], y_pred: List[int], y_prob: List[float]) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics."""

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # AUC score (handle case where all labels are the same)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5  # Random classifier performance

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            confusion_matrix=cm,
            classification_report=report
        )

    def import_model_from_path(self, model_path: Path, phase_name: str) -> 'PhaseDetectionModel':
        # Import PhaseDetectionModel here to avoid circular imports
        from ..detection import PhaseDetectionModel

        # Create and load the model
        config_path = str(model_path).replace('.pth', '_config.json')
        with open (config_path, 'r') as f:
            config = json.load(f)
        model_config = ModelConfig(**_filter_to_model_config_fields(config))
        model = PhaseDetectionModel(config=model_config, target_phase=phase_name)
        model.load_model(str(model_path))
        return model

    def import_models_from_path(self, model_path: Path, phase_name: str) -> 'PhaseDetectionModel':
        # Import PhaseDetectionModel here to avoid circular imports
        from ..detection import PhaseDetectionModel
        models = []

        # Create and load the models
        for solo_model_path in model_path:
            config_path = str(solo_model_path).replace('.pth', '_config.json')
            with open (config_path, 'r') as f:
                config = json.load(f)
            model_config = ModelConfig(**_filter_to_model_config_fields(config))
            model = PhaseDetectionModel(config=model_config, target_phase=phase_name)
            model.load_model(str(solo_model_path))
            models.append(model)
        return models

    def evaluate_model_on_dataset(self, phase_name: str, test_dir: str, ground_truth_file: Optional[str] = None) -> EvaluationMetrics:
        """
        Evaluate a single model on a test dataset.

        Args:
            phase_name: Name of the phase/model to evaluate
            test_dir: Directory containing test patterns
            ground_truth_file: Path to ground truth JSON file

        Returns:
            EvaluationMetrics object
        """
        if phase_name not in self.available_models:
            raise ValueError(f"Model for phase {phase_name} not found")

        # Load the actual trained model
        model_path = self.available_models[phase_name]
        print(f"Loading model: {model_path}")

        model = self.import_model_from_path(model_path, phase_name)

        print(f"Model loaded successfully for {phase_name}")

        # Load test patterns
        patterns, filenames = self.load_patterns_from_directory(test_dir)

        if not patterns:
            raise ValueError(f"No patterns found in {test_dir}")

        print(f"Loaded {len(patterns)} test patterns")

        print(f"Expected input size: {model.config.input_size}")

        # Regularize every pattern onto the grid this model was trained on, the
        # same way evaluate_experimental_patterns() does.
        #
        # load_patterns_from_directory() returns (N, 2) [2theta, intensity]
        # arrays, while the network takes a 1-D intensity trace plus a mask
        # channel when use_mask is set, resampled onto the model's own angular
        # range and point count.
        patterns = [
            regularize_input(
                file_name=filename,
                pattern=pattern,
                min_angle=model.config.min_angle,
                max_angle=model.config.max_angle,
                target_length=model.config.input_size,
                use_mask=model.config.use_mask,
                model_config=model.config,
            )
            for pattern, filename in zip(patterns, filenames)
        ]

        # Load ground truth
        if ground_truth_file and os.path.exists(ground_truth_file):
            ground_truth = self.load_ground_truth(ground_truth_file)
            labels = self.extract_labels_from_ground_truth(ground_truth, filenames, phase_name)
        else:
            # Fallback: infer labels from filenames
            labels = [1 if phase_name in fname else 0 for fname in filenames]

        # Make actual predictions using the loaded model
        predictions = []
        probabilities = []
        failed = []

        print(f"Making predictions on {len(patterns)} patterns...")
        for i, pattern in enumerate(patterns):
            try:
                # Get prediction probability
                prob = model.predict(pattern)
                probabilities.append(float(prob))

                # Convert to binary prediction (threshold = 0.5)
                pred = 1 if prob > 0.5 else 0
                predictions.append(pred)

            except Exception as e:
                print(f"Warning: Prediction failed for pattern {i}: {e}")
                failed.append(f"{filenames[i]}: {type(e).__name__}: {e}")
                predictions.append(0)  # Default to negative
                probabilities.append(0.0)

        # Defaulting a failed prediction to negative is reasonable for the
        # occasional unreadable pattern, but if every prediction failed the
        # metrics below would describe the error path rather than the model.
        if failed and len(failed) == len(patterns):
            raise RuntimeError(
                f"Every one of the {len(patterns)} predictions failed for "
                f"{phase_name!r}; the reported metrics would describe the error "
                f"path rather than the model. First failure: {failed[0]}"
            )
        if failed:
            print(f"Warning: {len(failed)} of {len(patterns)} predictions failed "
                  f"and were scored as negative.")

        print(f"Predictions completed. Positive predictions: {sum(predictions)}/{len(predictions)}")

        # Calculate metrics
        metrics = self.calculate_metrics(labels, predictions, probabilities)

        return metrics

    def create_evaluation_plots(self, metrics: EvaluationMetrics, phase_name: str, output_path: str):
        """Create evaluation plots."""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Confusion Matrix
        sns.heatmap(metrics.confusion_matrix, annot=True, fmt='d',
                   cmap='Blues', ax=axes[0])
        axes[0].set_title(phase_name, fontsize=14)
        axes[0].set_xlabel('Predicted', fontsize=12, labelpad=16)
        axes[0].set_ylabel('Actual', fontsize=12, labelpad=16)
        axes[0].tick_params(labelsize=11)

        # Metrics Bar Plot
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        metric_values = [metrics.accuracy, metrics.precision, metrics.recall,
                        metrics.f1_score, metrics.auc_score]

        axes[1].bar(metric_names, metric_values, color=['skyblue', 'lightgreen',
                                                       'lightcoral', 'gold', 'plum'])
        axes[1].set_title(phase_name, fontsize=14)
        axes[1].set_ylabel('Score', fontsize=12, labelpad=16)
        axes[1].set_xlabel('Metrics', fontsize=12, labelpad=16)
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(labelsize=11)

        # Add value labels on bars
        for i, v in enumerate(metric_values):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_comprehensive_evaluation(self, test_datasets: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, EvaluationMetrics]]:
        """
        Run comprehensive evaluation on multiple test datasets.

        Args:
            test_datasets: Dictionary mapping dataset names to {test_dir, ground_truth_file}

        Returns:
            Dictionary of evaluation results
        """
        results = {}

        for phase_name in self.available_models:
            results[phase_name] = {}

            print(f"\n{'='*60}")
            print(f"Evaluating {phase_name} model")
            print(f"{'='*60}")

            # Only evaluate on datasets that match this phase
            matching_datasets = {
                dataset_name: dataset_info
                for dataset_name, dataset_info in test_datasets.items()
                if phase_name in dataset_name
            }

            if not matching_datasets:
                print(f"No matching test datasets found for {phase_name}")
                continue

            for dataset_name, dataset_info in matching_datasets.items():
                print(f"\nDataset: {dataset_name}")
                print(f"Test directory: {dataset_info['test_dir']}")

                try:
                    # Run evaluation
                    metrics = self.evaluate_model_on_dataset(
                        phase_name=phase_name,
                        test_dir=dataset_info['test_dir'],
                        ground_truth_file=dataset_info.get('ground_truth_file')
                    )

                    results[phase_name][dataset_name] = metrics

                    # Print summary
                    print(f"Results for {phase_name} on {dataset_name}:")
                    print(f"  Accuracy:  {metrics.accuracy:.4f}")
                    print(f"  Precision: {metrics.precision:.4f}")
                    print(f"  Recall:    {metrics.recall:.4f}")
                    print(f"  F1-Score:  {metrics.f1_score:.4f}")
                    print(f"  AUC:       {metrics.auc_score:.4f}")

                    # Create plots with simplified filename and title
                    plot_path = self.output_dir / f"{phase_name}.png"
                    self.create_evaluation_plots(metrics, phase_name, str(plot_path))

                    # Save detailed results
                    results_path = self.output_dir / f"{phase_name}_{dataset_name}_metrics.json"
                    with open(results_path, 'w') as f:
                        json.dump(metrics.to_dict(), f, indent=2)

                except Exception as e:
                    print(f"  Error evaluating {phase_name} on {dataset_name}: {e}")
                    continue

        # Save comprehensive results
        self.save_comprehensive_results(results)

        return results

    def save_comprehensive_results(self, results: Dict[str, Dict[str, EvaluationMetrics]]):
        """Save comprehensive evaluation results."""

        # Create summary DataFrame
        summary_data = []

        for phase_name, dataset_results in results.items():
            for dataset_name, metrics in dataset_results.items():
                summary_data.append({
                    'Phase': phase_name,
                    'Dataset': dataset_name,
                    'Accuracy': metrics.accuracy,
                    'Precision': metrics.precision,
                    'Recall': metrics.recall,
                    'F1_Score': metrics.f1_score,
                    'AUC': metrics.auc_score
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # Save CSV
            csv_path = self.output_dir / "evaluation_summary.csv"
            summary_df.to_csv(csv_path, index=False)

            print(f"\n✓ Comprehensive evaluation results saved to {self.output_dir}")
            print(f"✓ Summary CSV: {csv_path}")

    def visualize_filters(self, model: torch.nn.Module):
        """
        Visualize all Conv1d filters in a DetectionCNN model.
        Each filter is plotted as a line plot.

        Args:
            model: instance of DetectionCNN
        """
        conv_count = 0

        for block_idx, block in enumerate(model.conv_layers):
            for layer in block:
                if isinstance(layer, torch.nn.Conv1d):
                    weights = layer.weight.data.cpu().numpy()  # shape: (out_channels, in_channels, kernel_size)
                    out_channels, in_channels, kernel_size = weights.shape

                    print(f"Block {block_idx} - Conv Layer {conv_count}: shape = {weights.shape}")

                    fig, axes = plt.subplots(out_channels, in_channels, figsize=(in_channels*2, out_channels*2))

                    # Ensure axes is always 2D array
                    if out_channels == 1 and in_channels == 1:
                        axes = np.array([[axes]])
                    elif out_channels == 1:
                        axes = np.array([axes])
                    elif in_channels == 1:
                        axes = np.array([[ax] for ax in axes])

                    for i in range(out_channels):
                        for j in range(in_channels):
                            ax = axes[i][j]
                            ax.plot(weights[i, j], color='blue')
                            ax.set_title(f"Out {i}, In {j}")
                            ax.set_xticks([])
                            ax.set_yticks([])

                    plt.suptitle(f"Conv Layer {conv_count} Filters (Block {block_idx})")
                    plt.tight_layout()
                    plt.show()

                    conv_count += 1

    def evaluate_experimental_patterns(self,
                                       exp_patterns_dir: str,
                                       group_phases: bool = False,
                                       group_similarity_threshold: Optional[float] = None,
                                       enable_chemistry_prefilter: bool = False,
                                       grouping_backend: str = "gaussian_cosine",
                                       probability_threshold: float = 0.5,
                                       top_k_config: int = None,
                                       dara_dict: Optional[dict] = None,
                                       pinned_phases_dir: Optional[str|Path] = None,
                                       plot_dir: Optional[str] = None) -> Dict[str, Dict[str, List[float]]]:
        """
        Evaluate models on experimental patterns organized by phase count.

        Args:
            exp_patterns_dir: Directory containing experimental patterns
            probability_threshold: Threshold for converting probabilities to binary predictions (default: 0.5)
            top_k_config: Configuration for top k scores predicted as positive
            plot_dir: Optional directory to save plots of processed patterns
            enable_chemistry_prefilter: passed through to
                `build_phase_groups_from_peaks` (default False -- grouping
                is profile-similarity only, matching DARA's own grouping).
            grouping_backend: passed through to
                `build_phase_groups_from_peaks` ("gaussian_cosine" default,
                the same similarity+clustering code DARA's own internal
                grouping uses via `DaraConfig.grouping_metric`; "legacy_bfs"
                kept only for historical comparison).

        Returns:
            Dictionary with predictions for each model on experimental data
        """
        exp_results = {}
        exp_path = Path(exp_patterns_dir)
        visualize_filters = True
        weight_extraction_failures = 0

        # Load all models once at the beginning
        loaded_models = {}
        model_input_size = None
        use_mask = None

        print("Loading all trained models...")
        time_start = time.time()
        failed_to_load = []
        for phase_name, model_path in self.available_models.items():
            last_err = None
            for attempt in range(1, 4):
                try:
                    if self.use_ensemble:
                        models = self.import_models_from_path(model_path, phase_name)
                        loaded_models[phase_name] = models
                        config_path = str(model_path[0]).replace('.pth', '_config.json')

                    else:
                        model = self.import_model_from_path(model_path, phase_name)
                        loaded_models[phase_name] = model
                        config_path = str(model_path).replace('.pth', '_config.json')

                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            model_expected_input_size = config.get('input_size', 4501)
                            model_use_mask = config.get('use_mask', False)

                        if model_input_size is None or use_mask is None:
                            model_input_size = model_expected_input_size
                            use_mask = model_use_mask
                        else:
                            if model_input_size != model_expected_input_size:
                                raise ValueError(f"Inconsistent model input sizes for {phase_name} model")
                            if use_mask != model_use_mask:
                                raise ValueError(f"Inconsistent use_mask for {phase_name} model")

                    print(f"  ✓ Loaded model for {phase_name} (input size: {model_input_size}, use_mask: {use_mask})")
                    print(f"  Model device: {model.device}")
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    is_cuda_oom = "CUDA" in str(e) and "out of memory" in str(e)
                    if is_cuda_oom and attempt < 3:
                        torch.cuda.empty_cache()
                        print(f"  ! CUDA OOM loading {phase_name} (attempt {attempt}/3) -- likely GPU "
                              f"contention from another process, retrying after backoff...")
                        time.sleep(5 * attempt)
                        continue
                    print(f"  ✗ Failed to load model for {phase_name}: {e}")
                    break
            if last_err is not None:
                failed_to_load.append(phase_name)

        time_end = time.time()
        print(f"All models loaded in {time_end - time_start:.2f} seconds.\n")

        if failed_to_load:
            # A partial phase set changes which candidates every downstream
            # pattern can even be scored against, so it must never pass as a
            # clean run. Partial loads typically indicate GPU memory contention
            # and are easy to mistake for model nondeterminism.
            raise RuntimeError(
                f"Failed to load {len(failed_to_load)}/{len(self.available_models)} phase models "
                f"after retries (likely GPU memory contention from other processes on this shared "
                f"GPU) -- refusing to silently evaluate with a partial phase set. "
                f"Missing: {sorted(failed_to_load)}"
            )

        # Generate a clean phase mapping dictionary ---
        known_sgs = set()
        phase_mapping = {}  # Map original_name -> cleaned_name

        for p in self.available_models.keys():
            clean_p = p.replace('_mp', '')
            p_parts = clean_p.split('_')

            if len(p_parts) > 1 and p_parts[1].isdigit():
                known_sgs.add(p_parts[1])

            # Define what a 'cleaned' name is (Formula_SG)
            if len(p_parts) >= 2 and p_parts[1].isdigit():
                cleaned_name = f"{p_parts[0]}_{p_parts[1]}"
            else:
                cleaned_name = p_parts[0]

            phase_mapping[p] = cleaned_name

        # Simply reading patterns [[2θ], [intensity]] from the directory, no further processing
        patterns, filenames = self.load_patterns_from_directory(
            str(exp_path),
            file_extension="xy"
        )

        exp_results = {}
        dara_results = {}


        print(
            f"Grouping highly similar phases "
            f"(similarity threshold: {group_similarity_threshold})..."
        )

        # Catalog of which catalog phases are XRD-indistinguishable from each
        # other at this threshold, used below both to group each pattern's
        # raw CNN-positive candidates into one prediction per group, and to
        # map each pattern's true phase onto the same group for scoring.
        cnn_grouped_phase_catalog = build_phase_groups_from_peaks(
            self.models_dir,
            group_similarity_threshold,
            enable_chemistry_prefilter=enable_chemistry_prefilter,
            grouping_backend=grouping_backend,
            )

        # Process each pattern
        for i, (pattern, filename) in enumerate(zip(patterns, filenames)):

            # preprocess pattern: truncate, pad, mask, interp to input size
            intensity = regularize_input(file_name = filename,
                                         pattern = pattern,
                                         min_angle = self.xrd_config.min_angle,
                                         max_angle = self.xrd_config.max_angle,
                                         target_length = model_input_size,
                                         use_mask = use_mask,
                                         model_config = self.model_config)     # (1, N) or (2, N) if use_mask

            # Treating filename as a list of phases present in the pattern
            base_name = Path(filename).stem

            # Strip '_mp' so formulas and SGs align properly
            clean_base_name = base_name.replace('_mp', '')
            parts = clean_base_name.split('_')

            # Strip trailing duplicate index correctly to fetch the actual ground truths from the filename
            if parts and parts[-1].isdigit():
                if len(parts) >= 2 and parts[-2].isdigit():
                    parts.pop()
                elif parts[-1] not in known_sgs:
                    parts.pop()

            # Parse remaining parts into formulas and their respective SGs
            file_formulas = {}
            current_formula = None
            for part in parts:
                if not part.isdigit():
                    current_formula = part
                    if current_formula not in file_formulas:
                        file_formulas[current_formula] = []
                else:
                    if current_formula is not None:
                        file_formulas[current_formula].append(part)

            # Determine phases in pattern directly from parsed file name
            phases_in_pattern = []
            for formula, sgs in file_formulas.items():
                if not sgs:
                    phases_in_pattern.append(formula)
                else:
                    for sg in sgs:
                        phases_in_pattern.append(f"{formula}_{sg}")

            phases_in_pattern = list(set(phases_in_pattern))

            print('\n'+'='*40)
            print(f"Processing pattern: {base_name}")

            start_time = time.time()

            pattern_results = {
                'filename': filename,
                'phases_in_pattern': phases_in_pattern,
                'phase_mapping': phase_mapping,  # <-- NEW: Expose mapping to the output JSON
                'model_predictions': {},
                'true_labels': {},
                'probabilities': {},
                'std': {},
                'cnn_predicted_labels': {},
                'dara_predicted_labels': {},
                'time': 0.0,
                'cnn_time': 0.0,
                'dara_time': 0.0
            }

            if self.use_ensemble:
                phase_model_iter = loaded_models.items()
            else:
                phase_model_iter = {k: [v] for k, v in loaded_models.items()}.items()

            phase_probs = {}
            phase_stds = {}
            phase_true = {}

            for phase_name, models in phase_model_iter:
                # Compare the CLEANED model name against the CLEANED ground truth pattern formulas
                cleaned_phase_name = phase_mapping[phase_name]

                # 1. Try exact match first (e.g., TiO2_136 == TiO2_136)
                if cleaned_phase_name in phases_in_pattern:
                    true_label = 1
                else:
                    # 2. Fallback: if experimental pattern has no SG info, match by base formula
                    model_formula = cleaned_phase_name.split('_')[0]
                    true_label = 1 if model_formula in phases_in_pattern else 0

                phase_true[phase_name] = true_label

                probabilities = []

                try:
                    for model in models:
                        prob = model.predict(intensity)
                        probabilities.append(float(prob))

                    phase_probs[phase_name] = float(np.mean(probabilities))
                    phase_stds[phase_name] = float(np.std(probabilities)) if self.use_ensemble else 0.0

                except Exception as e:
                    print(f"    Warning: Prediction failed for {phase_name}: {e}")
                    phase_probs[phase_name] = 0.0
                    phase_stds[phase_name] = 0.0
                    import traceback
                    traceback.print_exc()

            cnn_predicted_labels = {}

            use_top_k = top_k_config.get('use_top_k', False) if top_k_config else False
            if use_top_k:
                k_ratio = top_k_config.get('k_ratio', None)
                k_value = int(k_ratio * len(phase_probs)) if k_ratio else 5
                lower_threshold = top_k_config.get('lower_threshold', 0.1)
                upper_threshold = top_k_config.get('upper_threshold', 0.5)

                # Ranking by probability
                sorted_phases = sorted(phase_probs.items(), key=lambda x: x[1], reverse=True)

                # Define sets
                top_k = {name for name, _ in sorted_phases[:k_value]}
                low = {name for name, prob in phase_probs.items() if prob < lower_threshold}
                high = {name for name, prob in phase_probs.items() if prob > upper_threshold}

                # Final selection:  (top_k or high) − low
                final_selected = (top_k | high) - low

                for phase in phase_probs:
                    cnn_predicted_labels[phase] = int(phase in final_selected)

            else:
                if probability_threshold is None:
                    probability_threshold = 0.5
                for phase in phase_probs:
                    cnn_predicted_labels[phase] = int(phase_probs[phase] > probability_threshold)

            cnn_end_time = time.time()
            pattern_results['cnn_time'] = cnn_end_time - start_time

            use_dara = dara_dict.get('use_dara', False) if dara_dict else False
            # "cnn_positive" (default): DARA only ever sees phases the CNN already
            # flagged as present, which is the mode the reported results use.
            # "all": bypass the CNN gate entirely and hand DARA the full reference
            # pool, for a true "DARA-only" ablation. Candidates get a flat score in that case
            # (there is no CNN-informed ranking to sort by), so DARA's own search
            # isn't biased toward CNN-favored candidates.
            candidate_source = (dara_dict.get('candidate_source', 'cnn_positive') if dara_dict else 'cnn_positive')
            refined_phase_weights = {}
            dara_predicted_labels = {}

            # get list of pinned phases from pinned pahses dir
            pinned_phases = list(pinned_phases_dir.glob("*.cif")) if pinned_phases_dir else []

            if use_dara:
                dara_config = DaraConfig(**{k: v for k, v in dara_dict.items() if k != 'candidate_source'}) if dara_dict else DaraConfig()

                if candidate_source == 'all':
                    sorted_candidates = [
                        {
                            "original_name": phase,
                            "cleaned_name": phase_mapping[phase],
                            "score": 1.0,
                            "cif": self.ref_dir / f"{phase}.cif"
                        }
                        for phase in cnn_predicted_labels
                    ]
                else:
                    sorted_candidates = []
                    for phase in cnn_predicted_labels:
                        if cnn_predicted_labels[phase] == 1:
                            sorted_candidates.append({
                                "original_name": phase,
                                "cleaned_name": phase_mapping[phase],
                                "score": phase_probs[phase],
                                "cif": self.ref_dir / f"{phase}.cif"
                            })

                    sorted_candidates.sort(key=lambda x: x["score"], reverse=True)

                # Unpack all 4 returned values
                dara_result, final_rwp, max_false_peak_intensity, dara_groups_original = run_dara_refinement(
                    pattern_path=exp_path / filename,
                    sorted_candidates=sorted_candidates,
                    pinned_phases=pinned_phases,
                    dara_config=dara_config
                )

                dara_groups_clean = []  # default when DARA yields no valid result for this pattern

                if dara_result is not None:
                    dara_results[base_name] = dara_result
                    # Rietveld weight fractions of the refined phases.
                    try:
                        refined_phase_weights = dara_result.refinement_result.get_phase_weights()
                    except Exception as _wf_err:
                        print(f"    Could not extract phase weights: {_wf_err}")
                        refined_phase_weights = {}
                        # Recorded on the pattern's own result entry (not just printed) so a
                        # reader of the saved JSON -- not just the console -- can tell this
                        # pattern's weight fractions are missing rather than genuinely zero.
                        pattern_results['dara_weight_extraction_failed'] = str(_wf_err)
                        weight_extraction_failures += 1
                    # Translate DARA's original_name groupings into cleaned_name groupings
                    dara_groups_clean = []
                    for group in dara_groups_original:
                        if isinstance(group, (list, tuple)):
                            clean_group = [phase_mapping.get(orig, orig) for orig in group]
                            dara_groups_clean.append(list(set(clean_group)))

                        elif isinstance(group, str):
                            clean_name = phase_mapping.get(group, group)
                            dara_groups_clean.append([clean_name])

                    dara_winning_set = set()
                    for item in dara_groups_original:
                        if isinstance(item, (list, tuple)):
                            dara_winning_set.update(item)
                        elif isinstance(item, str):
                            dara_winning_set.add(item)

                    dara_predicted_labels = {
                        phase: int(phase in dara_winning_set)
                        for phase in cnn_predicted_labels
                    }


                    if dara_config.save_refined_plot:
                        refined_plot_dir = self.output_dir / "dara_refined_plots"
                        if refined_plot_dir.exists() is False:
                            refined_plot_dir.mkdir(parents=True, exist_ok=True)
                        dara_result.visualize().write_html(refined_plot_dir / f"{base_name}_refined.html")
                        print(f"    Saved DARA refined plot to {refined_plot_dir / f'{base_name}_refined.html'}")
                        print(f'Refinement result: {refined_phase_weights}')

                    print(f"DARA refinement result: Rwp={final_rwp:.2f}%, Max False Peak Relative Intensity={max_false_peak_intensity:.4f}")
                else:
                    print("    DARA refinement did not yield a valid result for this pattern.")
                    dara_predicted_labels = {}
                    refined_phase_weights = {}

            pattern_results['dara_time'] = (time.time() - cnn_end_time) if use_dara else 0.0

            cnn_only_predicted_phases = [
                phase
                for phase, label in cnn_predicted_labels.items()
                if label == 1
            ]

            # This will reflect what the CNN originally found
            cnn_grouped_phases = group_phases_func(
                cnn_only_predicted_phases,
                cnn_grouped_phase_catalog
            )

            if use_dara:
                final_predictions = dara_predicted_labels
            else:
                final_predictions = cnn_predicted_labels

            pattern_results["cnn_grouped_phases"] = cnn_grouped_phases
            if use_dara:
                pattern_results["dara_grouped_phases"] = dara_groups_clean

            pattern_results['true_labels'] = phase_true
            pattern_results['cnn_predicted_labels'] = cnn_predicted_labels

            if use_dara:
                pattern_results['dara_predicted_labels'] = dara_predicted_labels
                pattern_results['weights'] = refined_phase_weights if refined_phase_weights else {}

            pattern_results['probabilities'] = phase_probs
            pattern_results['std'] = phase_stds if self.use_ensemble else {}
            pattern_results['model_predictions'] = final_predictions

            metric_true_phases = phases_in_pattern
            metric_pred_groups = dara_groups_clean if use_dara else cnn_grouped_phases
            metric_catalog = dara_groups_clean if use_dara else cnn_grouped_phase_catalog

            # If ground truth lacks space groups (no underscores), strip them from predictions
            if not any('_' in p for p in metric_true_phases):
                metric_pred_groups = [[p.split('_')[0] for p in group] for group in metric_pred_groups]
                metric_catalog = [[p.split('_')[0] for p in group] for group in metric_catalog]

            phase_metrics = calculate_pattern_f1_metrics(
                true_phases=metric_true_phases,
                predicted_groups=metric_pred_groups,
                grouping_catalog=metric_catalog
            )
            pattern_results['phase_metrics'] = phase_metrics

            end_time = time.time()
            time_taken = end_time - start_time
            pattern_results['time'] = time_taken

            exp_results[base_name] = pattern_results

            print(f"    Recall: {pattern_results['phase_metrics']['recall']:.4f}, Precision: {pattern_results['phase_metrics']['precision']:.4f}, F1: {pattern_results['phase_metrics']['f1_score']:.4f}")
            print(f"    Time taken: {time_taken:.2f} seconds")

        print(f"  Processed {len(filenames)} patterns in total")
        if weight_extraction_failures:
            print(f"  Warning: DARA phase-weight extraction failed for {weight_extraction_failures}/{len(filenames)} "
                  f"patterns (see 'dara_weight_extraction_failed' in the saved JSON for which ones)")

        exp_results_path = self.output_dir / f"experimental_evaluation.json"
        with open(exp_results_path, 'w') as f:
            json.dump(exp_results, f, indent=2)

        print(f"\n✓ Experimental evaluation results saved to {exp_results_path}")

        csv_path = str(exp_results_path).replace('.json', '_summary.csv')
        generate_experimental_summary_csv(exp_results, csv_path)

        if use_dara:
            return exp_results, dara_results
        else:
            return exp_results, None


    def _plot_processed_pattern(self, two_theta: np.ndarray, intensity: np.ndarray,
                               pattern_name: str, plot_dir: str) -> None:
        """
        Plot a processed experimental pattern and save as PNG.

        Args:
            two_theta: 2θ values
            intensity: Intensity values
            pattern_name: Name of the pattern (without extension)
            plot_dir: Directory to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(two_theta, intensity, 'b-', linewidth=1.0)
        plt.xlabel('2θ (degrees)')
        plt.ylabel('Intensity (normalized to 0-100)')
        plt.title(f'Processed XRD Pattern: {pattern_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(plot_dir, f"{pattern_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory



    def visualize(
        self,
        result,
        phase_weight_threshold: float = 1e-3,
        base_name: str = "refinement_result",
        diff_offset: bool = False):
        """Visualize the result from the refinement. It uses plotly as the backend engine."""
        colormap = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        plot_data = result.plot_data

        # Create a Plotly figure with size 800x600
        fig = go.Figure()

        # Adding scatter plot for observed data
        fig.add_trace(
            go.Scatter(
                x=plot_data.x,
                y=plot_data.y_obs,
                mode="markers",
                marker=dict(color="blue", size=3, symbol="cross-thin-open"),
                name="Observed",
            )
        )

        # Adding line plot for calculated data
        fig.add_trace(
            go.Scatter(
                x=plot_data.x,
                y=plot_data.y_calc,
                mode="lines",
                line=dict(color="green", width=2),
                name="Calculated",
            )
        )

        # Adding line plot for background
        fig.add_trace(
            go.Scatter(
                x=plot_data.x,
                y=plot_data.y_bkg,
                mode="lines",
                line=dict(color="#FF7F7F", width=2),
                name="Background",
                opacity=0.5,
            )
        )

        diff = np.array(plot_data.y_obs) - np.array(plot_data.y_calc)
        diff_offset_val = 1.1 * max(diff) if diff_offset else 0  # 10 percent below

        # Adding line plot for difference
        fig.add_trace(
            go.Scatter(
                x=plot_data.x,
                y=diff - diff_offset_val,
                mode="lines",
                line=dict(color="#808080", width=1),
                name="Difference",
                hoverinfo="skip",  # "skip" to not show hover info for this trace
                opacity=0.7,
            )
        )

        # if there is no phase weight, it will return an empty dictionary (not shown in the legend)
        try:
            weight_fractions = result.get_phase_weights()
        except TypeError:
            weight_fractions = {}

        peak_data = result.peak_data
        max_y = max(np.array(result.plot_data.y_obs) + np.array(result.plot_data.y_bkg))
        min_y_diff = min(
            np.array(result.plot_data.y_obs) - np.array(result.plot_data.y_calc)
        )
        # Adding dashed lines for phases
        for i, (phase_name, phase) in enumerate(plot_data.structs.items()):

            # Skipping phases below the weight threshold
            if weight_fractions[phase_name] < phase_weight_threshold:
                continue

            # add area under the curve between the curve and the plot_data["y_bkg"]
            if i >= len(colormap) - 1:
                i = i % (len(colormap) - 1)

            name = (
                f"{phase_name} ({weight_fractions[phase_name] * 100:.2f} %)"
                if len(weight_fractions) > 1
                else phase_name
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_data.x,
                    y=plot_data.y_bkg,
                    mode="lines",
                    line=dict(color=colormap[i], width=0),
                    fill=None,
                    showlegend=False,
                    hoverinfo="none",
                    legendgroup=phase_name,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_data.x,
                    y=np.array(phase) + np.array(plot_data.y_bkg),
                    mode="lines",
                    line=dict(color=colormap[i], width=1.5),
                    fill="tonexty",
                    name=name,
                    visible="legendonly",
                    legendgroup=phase_name,
                )
            )
            refl = peak_data[peak_data["phase"] == phase_name]["2theta"]
            intensity = peak_data[peak_data["phase"] == phase_name]["intensity"]
            fig.add_trace(
                go.Scatter(
                    x=refl,
                    y=np.ones(len(refl)) * (i + 1) * -max_y * 0.1 + min_y_diff,
                    mode="markers",
                    marker={
                        "symbol": 142,
                        "size": 5,
                        "color": colormap[i],
                    },
                    name=name,
                    legendgroup=phase_name,
                    showlegend=False,
                    visible="legendonly",
                    text=[f"{x:.2f}, {y:.2f}" for x, y in zip(refl, intensity)],
                    hovertemplate="%{text}",
                )
            )

        title = f"{result.lst_data.pattern_name} (Rwp={result.lst_data.rwp:.2f}%)"

        # Updating layout with titles and labels
        fig.update_layout(
            autosize=True,
            xaxis=dict(
                range=[min(plot_data.x), max(plot_data.x)],
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            ),
            title=title,
            xaxis_title="2θ [°]",
            yaxis_title="Intensity",
            legend_title="",
            font=dict(family="Arial, sans-serif", color="RebeccaPurple"),
            plot_bgcolor="white",
            yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
            legend_tracegroupgap=1,
        )

        fig.add_hline(y=0, line_width=1)

        # add tick
        fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor="black", ticklen=10)
        fig.update_yaxes(ticks="outside", tickwidth=1, tickcolor="black", ticklen=10)

        dir = self.output_dir / "refined_plots"
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory for refined plots: {dir}")
        fig.write_html(dir / f"{base_name}_refined_plot.html")

        return fig



def _strip_leading_zero_intensity(pattern_path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """
    Strip a leading run of exactly-zero intensity from an XY pattern before
    handing it to DARA. BGMN wastes computation (or, in some patterns, errors
    out) fitting peaks to a long flat zero-signal region at the start of a
    scan -- common when data collection starts well before the first real
    peak.

    DARA reads `pattern_path` directly off disk (it copies the raw file into
    its own BGMN working directory), so this has to produce a real file, not
    just an in-memory array.

    Returns (path_to_use, temp_dir_or_None): temp_dir is None (and
    path_to_use is the original path, unchanged) when there's no leading
    zero run to strip. Caller is responsible for calling temp_dir.cleanup()
    once DARA is done reading the file.
    """
    header_lines = count_header_lines(pattern_path)
    data = np.loadtxt(pattern_path, skiprows=header_lines)
    intensity = data[:, 1]
    nonzero = np.flatnonzero(intensity)

    if len(nonzero) == 0 or nonzero[0] == 0:
        return pattern_path, None

    temp_dir = tempfile.TemporaryDirectory(prefix="galaxi_dara_pattern_")
    cleaned_path = Path(temp_dir.name) / pattern_path.name
    np.savetxt(cleaned_path, data[nonzero[0]:], header="2theta intensity")
    return cleaned_path, temp_dir


def run_dara_refinement(
    pattern_path: Path,
    sorted_candidates: List[Dict[str, Union[str, float, Path]]],
    pinned_phases: List,
    dara_config: DaraConfig,
):
    """
    Single-batch DARA refinement using all CNN-predicted phase candidates.
    """

    # Extract the CIF paths from our new dictionary structure
    batch_phases = [candidate["cif"] for candidate in sorted_candidates]

    print(f"\nDARA Single Batch Refinement: Refining {len(batch_phases)} phases:")

    cleaned_pattern_path, cleanup_dir = _strip_leading_zero_intensity(pattern_path)
    try:
        # Run DARA refinement on ALL candidates simultaneously
        refined_results = search_phases(
            pattern_path=cleaned_pattern_path,
            downsized_length=dara_config.downsized_length,
            phases=batch_phases,
            pinned_phases=pinned_phases,
            max_phases=4,
            wavelength="Cu",
            instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
            express_mode=True,
            enable_angular_cut=dara_config.enable_angular_cut,
            maximum_grouping_distance=dara_config.maximum_grouping_distance,
            grouping_metric=dara_config.grouping_metric,
            phase_params=dara_config.phase_params or {},
            refinement_params=dara_config.refinement_params or {},
            return_search_tree=True,
            record_peak_matcher_scores=True,
            score_coefficients=dara_config.score_coefficients,
            strike_threshold=dara_config.strike_threshold,
            rpb_threshold=dara_config.rpb_threshold,
            overfitting_threshold=dara_config.overfitting_threshold,
            false_peak_threshold=dara_config.false_peak_threshold,
            strain_threshold=dara_config.strain_threshold,
            early_stopping=dara_config.early_stopping,
        )
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()

    if dara_config.show_search_tree:
        print(f'Search tree:')
        refined_results.show(stdout=True, idhidden=True)

    result_tmp = refined_results.get_search_results()

    if not result_tmp:
        return None, None, None, None  # <-- Return 4 items on failure

    result = result_tmp[0]

    dara_groups_original = []

    try:
        for phase_tuple in result.phases:
            current_group = []
            for phase_obj in phase_tuple:
                if phase_obj.path:
                    current_group.append(
                        phase_obj.path.stem
                    )

            if current_group:
                dara_groups_original.append(
                    sorted(current_group)
                )

    except Exception as e:
        print(
            f"Warning: Could not parse "
            f"DARA winning phases: {e}"
        )
        dara_groups_original = []

    print(f"DARA refinement selected groups:")

    for g in dara_groups_original:
        print(g)

    refinement = result.refinement_result
    rwp = refinement.lst_data.rwp
    max_intensity = max(result.refinement_result.plot_data.y_obs)

    # Calculate false peak intensities safely
    max_extra_peak_intensity = max(result.extra_peaks, key=lambda x: x[1])[1] / max_intensity if result.extra_peaks else 0
    max_missing_peak_intensity = max(result.missing_peaks, key=lambda x: x[1])[1] / max_intensity if result.missing_peaks else 0
    max_false_peak_intensity = max(max_extra_peak_intensity, max_missing_peak_intensity)

    return result, rwp, max_false_peak_intensity, dara_groups_original
