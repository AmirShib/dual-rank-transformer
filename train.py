import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup,
    HfArgumentParser
)
from datasets import load_from_disk

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

from src.modeling import UniversalDualRanking
from src.data import DualMultipleChoiceCollator
from src.utils import CheckpointManager, EarlyStopping, ExperimentLogger

@dataclass
class TrainingConfig:
    """
    Configuration arguments for the dual-task ranking training pipeline.
    
    Attributes:
        model_name: HuggingFace model identifier or path to pre-trained model.
        data_dir: Path to the directory containing processed 'train' and 'val' datasets.
        output_dir: Directory where checkpoints and logs will be saved.
        mixed_precision: Precision mode ('no', 'fp16', 'bf16') for efficiency.
        use_margin: Whether to include the pairwise margin loss component.
        margin_weight: Weight scaling for the margin loss term relative to cross-entropy.
    """
    model_name: str = field(default="onlplab/alephbert-base", metadata={"help": "Pretrained model path"})
    data_dir: str = field(default="data_cache", metadata={"help": "Directory containing datasets"})
    output_dir: str = field(default="experiments", metadata={"help": "Artifact output directory"})
    
    batch_size: int = field(default=16, metadata={"help": "Batch size per device"})
    lr: float = field(default=2e-5, metadata={"help": "Learning rate"})
    epochs: int = field(default=20, metadata={"help": "Max training epochs"})
    use_margin: bool = field(default=True, metadata={"help": "Use composite margin loss"})
    margin_weight: float = field(default=0.5, metadata={"help": "Weight for margin loss"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    
    patience: int = field(default=5, metadata={"help": "Early stopping patience"})
    save_total_limit: int = field(default=2, metadata={"help": "Max checkpoints to keep"})
    metric_for_best: str = field(default="val_loss", metadata={"help": "Metric for model selection"})
    
    use_accelerate: bool = field(default=True, metadata={"help": "Use Hugging Face Accelerator"})
    mixed_precision: str = field(default="fp16", metadata={"help": "fp16 or bf16"})
    num_workers: int = field(default=4, metadata={"help": "DataLoader workers"})

def set_seed(seed: int):
    """Enforce reproducibility across numpy, torch, and cuda."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Computes ranking metrics: Top-1 Accuracy and Mean Reciprocal Rank (MRR).
    
    Args:
        logits: Raw scores from the model [Batch, Candidates].
        labels: Ground truth indices [Batch].
    """
    if logits is None or labels is None:
        return {}
        
    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    hits = (sorted_indices == labels.view(-1, 1))
    ranks = (hits.nonzero(as_tuple=False)[:, 1].float() + 1.0)
    
    return {
        "acc": (ranks == 1).float().mean().item(),
        "mrr": (1.0 / ranks).mean().item()
    }

def train_one_epoch(
    epoch_index: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    accelerator: Optional[Any],
    device: torch.device
) -> float:
    """
    Executes a single training epoch, handling forward/backward passes and gradient updates.
    
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    
    # Disable progress bars on non-main processes to avoid log clutter
    disable_tqdm = (accelerator is not None and not accelerator.is_main_process)
    pbar = tqdm(dataloader, desc=f"Ep {epoch_index} Train", disable=disable_tqdm)
    
    for batch in pbar:
        # Manually move to device if Accelerator is not handling placement
        if not accelerator: 
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        if accelerator:
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
        else:
            loss.backward()
            # Gradient clipping is crucial for Transformer stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / len(dataloader)

def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    accelerator: Optional[Any],
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates the model on the validation set.
    
    Returns:
        Dictionary containing aggregated loss, MRR, and Accuracy for both tasks.
    """
    model.eval()
    total_loss = 0.0
    val_metrics = {"occ_acc": [], "occ_mrr": [], "sec_acc": [], "sec_mrr": []}
    
    disable_tqdm = (accelerator is not None and not accelerator.is_main_process)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", disable=disable_tqdm):
            if not accelerator: 
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = model(**batch)
            
            # Aggregate loss across all GPUs for accurate logging
            current_loss = outputs.loss
            if accelerator:
                current_loss = accelerator.gather(current_loss).mean()
            total_loss += current_loss.item()
            
            # Compute metrics on CPU to conserve GPU memory
            if outputs.occ_logits is not None:
                m = calculate_metrics(outputs.occ_logits.cpu(), batch["occ_labels"].cpu())
                val_metrics["occ_acc"].append(m["acc"])
                val_metrics["occ_mrr"].append(m["mrr"])
                
            if outputs.sec_logits is not None:
                m = calculate_metrics(outputs.sec_logits.cpu(), batch["sector_labels"].cpu())
                val_metrics["sec_acc"].append(m["acc"])
                val_metrics["sec_mrr"].append(m["mrr"])

    return {
        "val_loss": total_loss / len(dataloader),
        "occ_mrr": np.mean(val_metrics["occ_mrr"]) if val_metrics["occ_mrr"] else 0.0,
        "sec_mrr": np.mean(val_metrics["sec_mrr"]) if val_metrics["sec_mrr"] else 0.0,
        "occ_acc": np.mean(val_metrics["occ_acc"]) if val_metrics["occ_acc"] else 0.0,
        "sec_acc": np.mean(val_metrics["sec_acc"]) if val_metrics["sec_acc"] else 0.0,
    }

def run_training(config: TrainingConfig):
    """
    Orchestrates the full training lifecycle including distributed setup, 
    data loading, model initialization, and the training loop.
    """
    set_seed(config.seed)
    
    accelerator = None
    if config.use_accelerate and ACCELERATE_AVAILABLE:
        accelerator = Accelerator(mixed_precision=config.mixed_precision)
        device = accelerator.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize logger only on main process to prevent duplicate file writes
    logger = None
    if accelerator is None or accelerator.is_main_process:
        logger = ExperimentLogger(config.output_dir, config)
        logger.info(f"Starting training on {device}")

    # Standard PyTorch format required for the model
    ds_train = load_from_disk(os.path.join(config.data_dir, "train"))
    ds_val = load_from_disk(os.path.join(config.data_dir, "val"))
    ds_train.set_format("torch")
    ds_val.set_format("torch")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    collator = DualMultipleChoiceCollator(tokenizer)

    train_loader = DataLoader(
        ds_train, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collator=collator,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        ds_val, 
        batch_size=config.batch_size, 
        collator=collator, 
        num_workers=config.num_workers
    )

    model = UniversalDualRanking.from_pretrained(
        config.model_name, 
        margin_weight=config.margin_weight
    )
    if not accelerator: 
        model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr)
    num_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_steps * 0.1), num_training_steps=num_steps
    )

    if accelerator:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

    # Setup MLOps managers (Checkpointing & Early Stopping)
    checkpoint_manager = None
    early_stopper = None
    if accelerator is None or accelerator.is_main_process:
        checkpoint_manager = CheckpointManager(
            config.output_dir, accelerator, 
            metric_name=config.metric_for_best, 
            save_total_limit=config.save_total_limit
        )
        early_stopper = EarlyStopping(patience=config.patience, mode="min")

    for epoch in range(1, config.epochs + 1):
        avg_train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, scheduler, accelerator, device
        )

        metrics = validate(model, val_loader, accelerator, device)
        
        should_stop = False
        
        # Log results and manage checkpoints on the main process
        if accelerator is None or accelerator.is_main_process:
            metrics["train_loss"] = avg_train_loss
            
            logger.log_epoch(epoch, metrics)
            logger.info(f"Ep {epoch} Results: {metrics}")

            current_score = metrics.get(config.metric_for_best, metrics["val_loss"])
            checkpoint_manager.save_checkpoint(epoch, current_score, model, tokenizer)
            
            early_stopper(current_score)
            if early_stopper.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                should_stop = True

        # Broadcast the early stopping decision to all GPUs to ensure synchronized exit
        if accelerator:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                break
        elif should_stop:
            break

    if logger: logger.info("Training complete.")

if __name__ == "__main__":
    parser = HfArgumentParser(TrainingConfig)
    config = parser.parse_args_into_dataclasses()[0]
    run_training(config)
