"""
Base classes for unified model interfaces.
"""

import copy
import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from .config import ModelConfig, DEFAULT_MODEL_CONFIG
from .structures import StructureManager


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self,
                 reference_dir: str = "References",
                 config: Optional[ModelConfig] = None):
        """
        Initialize base model.

        Args:
            reference_dir: Directory containing reference CIF files
            config: Model configuration
        """
        # Deep-copy rather than alias: subclasses may mutate self.config's
        # fields in place, and aliasing the DEFAULT_MODEL_CONFIG singleton (or
        # a config object shared by another instance) would leak that mutation
        # across every instance that didn't get its own copy.
        self.config = copy.deepcopy(config) if config is not None else copy.deepcopy(DEFAULT_MODEL_CONFIG)
        self.structure_manager = StructureManager(reference_dir)
        self.model = None
        self.training_history = None

    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, pattern: np.ndarray) -> Any:
        """Make predictions on a pattern."""
        pass

    def load_model(self, model_path: str):
        """Load pre-trained weights into `self.model`.

        `self.model` must already be constructed (e.g. by a subclass's own
        setup/train step) before calling this: only a state_dict is stored on
        disk and loaded here, not a pickled module object, so this is safe to
        call on model files from an untrusted source.
        """
        if self.model is None:
            raise ValueError(
                "self.model must be constructed before load_model() can load its weights"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def save_model(self, model_path: str):
        """Save the trained model's weights (state_dict only)."""
        if self.model is None:
            raise ValueError("No trained model to save")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def get_available_phases(self) -> List[str]:
        """Get list of available phase names."""
        return self.structure_manager.get_phase_names()
