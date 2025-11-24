"""
Experiment tracking and logging utilities.

Provides structured logging, experiment tracking, and result management.
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Tracks experiments with logging, metrics, and results management.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "experiments",
        use_tensorboard: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Root directory for experiment logs
            use_tensorboard: Whether to use TensorBoard logging
            config: Configuration dictionary to save
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Create experiment directory with timestamp
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(log_dir) / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_file = self.experiment_dir / "experiment.log"
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.config_file = self.experiment_dir / "config.yaml"
        
        # Initialize TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard"))
            logger.info(f"TensorBoard logging enabled: {self.experiment_dir / 'tensorboard'}")
        else:
            self.tb_writer = None
            if use_tensorboard and not TENSORBOARD_AVAILABLE:
                logger.warning("TensorBoard requested but not available")
        
        # Save configuration
        if config:
            from .config import save_config
            save_config(config, str(self.config_file))
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step/epoch
        """
        self.metrics[name].append({
            'value': float(value),
            'step': step,
            'timestamp': time.time()
        })
        
        # Update best metrics
        if name not in self.best_metrics or value > self.best_metrics[name]['value']:
            self.best_metrics[name] = {
                'value': float(value),
                'step': step,
                'timestamp': time.time()
            }
        
        # TensorBoard logging
        if self.tb_writer and step is not None:
            self.tb_writer.add_scalar(name, value, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log hyperparameters and optionally associated metrics.
        
        Args:
            hparams: Hyperparameter dictionary
            metrics: Optional metrics dictionary
        """
        if self.tb_writer:
            # TensorBoard expects flat dictionaries
            flat_hparams = self._flatten_dict(hparams)
            self.tb_writer.add_hparams(flat_hparams, metrics or {})
    
    def save_metrics(self) -> None:
        """Save metrics to JSON file."""
        metrics_data = {
            'metrics': dict(self.metrics),
            'best_metrics': self.best_metrics,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Saved metrics to: {self.metrics_file}")
    
    def get_metric_history(self, name: str) -> List[Dict[str, Any]]:
        """
        Get history of a specific metric.
        
        Args:
            name: Metric name
            
        Returns:
            List of metric records
        """
        return self.metrics.get(name, [])
    
    def get_best_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get best value of a specific metric.
        
        Args:
            name: Metric name
            
        Returns:
            Best metric record or None
        """
        return self.best_metrics.get(name)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """
        Log text to TensorBoard.
        
        Args:
            tag: Text tag
            text: Text content
            step: Training step/epoch
        """
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
    
    def close(self) -> None:
        """Close the experiment tracker and save final results."""
        self.save_metrics()
        
        if self.tb_writer:
            self.tb_writer.close()
        
        # Save summary
        duration = datetime.now() - self.start_time
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': duration.total_seconds(),
            'best_metrics': self.best_metrics
        }
        
        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Experiment completed. Duration: {duration}")
        logger.info(f"Results saved to: {self.experiment_dir}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
        """Flatten nested dictionary for TensorBoard."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to simple types for TensorBoard
                if isinstance(v, (int, float, str, bool)):
                    items.append((new_key, v))
                else:
                    items.append((new_key, str(v)))
        return dict(items)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional log file path
        level: Logging level
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )
