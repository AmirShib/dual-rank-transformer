import os
import json
import time
import shutil
import torch
import logging
from dataclasses import asdict
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """
    Handles atomic JSON logging to ensure data integrity even if the process crashes.
    """
    def __init__(self, log_dir: str, config: Any):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"training_log_{int(time.time())}.json")
        self.data = {
            "config": asdict(config),
            "system_info": {
                "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            },
            "history": []
        }
        self._flush()

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        entry = {
            "epoch": epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **metrics
        }
        self.data["history"].append(entry)
        self._flush()

    def _flush(self):
        # Write to temp file then rename to avoid partial writes
        temp_path = self.log_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(temp_path, self.log_path)
        
    def info(self, msg: str):
        print(f"[INFO] {msg}")

class EarlyStopping:
    """
    Monitors a metric and stops training if it doesn't improve after `patience` epochs.
    """
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Adjust score function based on optimization direction
        if mode == 'min':
            self.val_score_fn = lambda x: -x 
        else:
            self.val_score_fn = lambda x: x

    def __call__(self, current_score):
        score = self.val_score_fn(current_score)

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class CheckpointManager:
    """
    Manages model artifacts, keeping only the 'best' model and the last K checkpoints.
    Compatible with Hugging Face Accelerate.
    """
    def __init__(self, output_dir, accelerator, metric_name="val_loss", mode="min", save_total_limit=3):
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.metric_name = metric_name
        self.mode = mode
        self.save_total_limit = save_total_limit
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.saved_checkpoints = []

    def save_checkpoint(self, epoch, current_metric, model, tokenizer):
        # Save rolling checkpoint
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}")
        
        if self.accelerator:
            self.accelerator.save_state(ckpt_dir)
            unwrapped = self.accelerator.unwrap_model(model)
            unwrapped.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
        else:
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
        
        self.saved_checkpoints.append(ckpt_dir)
        
        # Enforce limit
        if len(self.saved_checkpoints) > self.save_total_limit:
            to_remove = self.saved_checkpoints.pop(0)
            if os.path.exists(to_remove):
                shutil.rmtree(to_remove)

        # Save Best Model
        is_best = (current_metric < self.best_metric) if self.mode == 'min' else (current_metric > self.best_metric)
        if is_best:
            self.best_metric = current_metric
            best_dir = os.path.join(self.output_dir, "best_model")
            
            if self.accelerator:
                unwrapped = self.accelerator.unwrap_model(model)
                unwrapped.save_pretrained(best_dir)
            else:
                model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
