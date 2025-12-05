"""
Training utilities for sentiment classification.

Provides:
- Training loop with validation
- Metrics calculation (accuracy, F1)
- Early stopping support
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1  # 10% of training steps for warmup
    max_grad_norm: float = 1.0
    device: str = "cpu"
    batch_size: int = 16  # For calculating total steps
    
    # Class weights for imbalanced data (optional)
    class_weights: Optional[List[float]] = None
    
    # Early stopping
    early_stopping_patience: int = 2


@dataclass
class TrainingResult:
    """Results from training."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    val_f1_scores: List[float] = field(default_factory=list)
    best_accuracy: float = 0.0
    best_f1: float = 0.0
    training_time: float = 0.0


class Trainer:
    """Trainer for sentiment classification models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = config.device
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer (only for trainable parameters)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup loss function
        if config.class_weights is not None:
            weights = torch.tensor(config.class_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Scheduler will be initialized when we know total steps
        self.scheduler = None
        self.total_steps = 0
        self.current_step = 0
    
    def _setup_scheduler(self, warmup_steps: int, total_steps: int) -> None:
        """Setup linear warmup + linear decay scheduler.
        
        Args:
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
        """
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Linear decay
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.current_step = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Update scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step()
                self.current_step += 1
            
            total_loss += loss.item()
            num_batches += 1
            
            # Show loss and LR in progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> Tuple[float, float, float, Dict]:
        """Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
        
        Returns:
            Tuple of (loss, accuracy, f1_score, classification_report_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        
        report = classification_report(
            all_labels, all_preds,
            target_names=["negative", "neutral", "positive"],
            output_dict=True,
        )
        
        return avg_loss, accuracy, f1, report
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> TrainingResult:
        """Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (overrides config if provided)
        
        Returns:
            TrainingResult with metrics history
        """
        epochs = epochs or self.config.epochs
        result = TrainingResult()
        
        start_time = time.time()
        
        # Calculate total steps and setup scheduler
        self.total_steps = len(train_loader) * epochs
        warmup_steps = int(self.total_steps * self.config.warmup_ratio)
        self._setup_scheduler(warmup_steps, self.total_steps)
        
        print(f"\nTraining for {epochs} epoch(s)...")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Total steps: {self.total_steps}, Warmup steps: {warmup_steps}")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        # Early stopping tracking
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            result.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc, val_f1, report = self.evaluate(val_loader)
            result.val_losses.append(val_loss)
            result.val_accuracies.append(val_acc)
            result.val_f1_scores.append(val_f1)
            
            # Track best
            if val_acc > result.best_accuracy:
                result.best_accuracy = val_acc
            if val_f1 > result.best_f1:
                result.best_f1 = val_f1
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_acc:.4f} ({val_acc*100:.1f}%)")
            print(f"  Val F1:     {val_f1:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        result.training_time = time.time() - start_time
        print("-" * 50)
        print(f"Training complete in {result.training_time:.1f}s")
        print(f"Best Val Accuracy: {result.best_accuracy*100:.1f}%")
        print(f"Best Val F1: {result.best_f1:.4f}")
        
        return result


def quick_evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Quick evaluation without full trainer setup.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
    
    Returns:
        Dictionary with accuracy and f1 score
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
    }