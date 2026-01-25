import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DualRankingOutput(ModelOutput):
    """
    Output container for the dual-task ranking model.

    Args:
        loss (Optional[torch.Tensor]): Combined loss for both tasks.
        occ_logits (Optional[torch.Tensor]): Logits for the occupation ranking task [B, K].
        sec_logits (Optional[torch.Tensor]): Logits for the sector ranking task [B, K].
    """
    loss: Optional[torch.Tensor] = None
    occ_logits: Optional[torch.Tensor] = None
    sec_logits: Optional[torch.Tensor] = None

class UniversalDualRanking(PreTrainedModel):
    """
    A multi-head ranking model that jointly optimizes for Occupation and Sector prediction
    using a composite loss (Listwise Cross-Entropy + Pairwise Margin).
    """
    def __init__(self, config, margin_weight: float = 0.5):
        super().__init__(config)
        self.margin_weight = margin_weight
        
        self.backbone = AutoModel.from_config(config)
        hidden_size = config.hidden_size
        head_dim = hidden_size // 2
        
        # Two-layer MLP heads with LayerNorm for better representation decoupling
        self.occ_head = nn.Sequential(
            nn.Linear(hidden_size, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Dropout(0.1),
            nn.Linear(head_dim, 1)
        )

        self.sec_head = nn.Sequential(
            nn.Linear(hidden_size, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Dropout(0.1),
            nn.Linear(head_dim, 1)
        )
        
        self.post_init()

    @staticmethod
    def soft_margin_weighted_loss(logits: torch.Tensor, labels: torch.Tensor, 
                                  mask: Optional[torch.Tensor] = None, 
                                  margin: float = 0.22, tau: float = 0.1) -> torch.Tensor:
        """
        Computes a soft margin loss weighted by the model's uncertainty gap between
        the top two candidates.
        """
        B, K = logits.shape
        if mask is None: 
            mask = torch.ones_like(logits, dtype=torch.bool)
        else: 
            mask = mask.to(dtype=torch.bool)

        # Select scores of the ground truth positives
        pos_score = logits.gather(dim=1, index=labels.view(B, 1)).squeeze(1)
        
        # Calculate uncertainty weight based on the gap between top-1 and top-2
        probs = logits.softmax(dim=-1)
        if K > 1:
            top2 = probs.topk(2, dim=-1).values
            gap = (top2[:, 0] - top2[:, 1])
            w_ex = 0.1 + 0.9 * torch.sigmoid((margin - gap) / 0.05)
        else:
            w_ex = torch.ones(B, device=logits.device)

        # Vectorized pairwise loss calculation
        is_pos = torch.zeros_like(mask).scatter_(1, labels.view(B, 1), True)
        neg_mask = mask & (~is_pos)
        
        diff = (pos_score.unsqueeze(1) - logits) / tau
        per_negative = F.softplus(-diff) * neg_mask.float()
        
        # Normalize by the number of valid negative candidates to prevent scaling issues
        mean_per_ex = per_negative.sum(dim=1) / neg_mask.float().sum(dim=1).clamp_min(1.0)
        
        return (mean_per_ex * w_ex).mean()

    def _extract_embeddings(self, input_ids, attention_mask, token_type_ids) -> Tuple[torch.Tensor, int, int]:
        """Flattens input for the backbone and extracts pooled representations."""
        B, K, T = input_ids.shape
        
        flat_input_ids = input_ids.view(-1, T)
        flat_attention_mask = attention_mask.view(-1, T) if attention_mask is not None else None
        
        model_inputs = {
            "input_ids": flat_input_ids,
            "attention_mask": flat_attention_mask
        }

        # Handle token_type_ids only if the config expects them (e.g., BERT vs RoBERTa)
        if token_type_ids is not None and getattr(self.config, "type_vocab_size", 0) > 0:
            model_inputs["token_type_ids"] = token_type_ids.view(-1, T)

        outputs = self.backbone(**model_inputs)
        
        # Standard CLS pooling (index 0)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings, B, K

    def _process_task(self, input_ids, attention_mask, token_type_ids, 
                      choice_mask, labels, head_module) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        if input_ids is None:
            return None, None

        reps, B, K = self._extract_embeddings(input_ids, attention_mask, token_type_ids)
        logits = head_module(reps).view(B, K)

        if choice_mask is not None:
            logits = logits.masked_fill(~choice_mask, float('-inf'))

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            margin_loss = self.soft_margin_weighted_loss(logits, labels, choice_mask)
            loss = ce_loss + (self.margin_weight * margin_loss)

        return logits, loss

    def forward(self, occ_input_ids=None, occ_labels=None, occ_choice_mask=None,
                sec_input_ids=None, sector_labels=None, sec_choice_mask=None, **kwargs):
        
        total_loss = torch.tensor(0.0, device=self.device)
        results = {}

        occ_logits, occ_loss = self._process_task(
            occ_input_ids, 
            kwargs.get('occ_attention_mask'), 
            kwargs.get('occ_token_type_ids'),
            occ_choice_mask, occ_labels, self.occ_head
        )
        if occ_logits is not None:
            results["occ_logits"] = occ_logits
            if occ_loss is not None: total_loss += occ_loss

        sec_logits, sec_loss = self._process_task(
            sec_input_ids, 
            kwargs.get('sec_attention_mask'), 
            kwargs.get('sec_token_type_ids'),
            sec_choice_mask, sector_labels, self.sec_head
        )
        if sec_logits is not None:
            results["sec_logits"] = sec_logits
            if sec_loss is not None: total_loss += sec_loss

        return DualRankingOutput(
            loss=total_loss if (occ_labels is not None or sector_labels is not None) else None,
            **results
        )
