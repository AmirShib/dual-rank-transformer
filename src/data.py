import torch
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizerBase

@dataclass
class DualMultipleChoiceCollator:
    """
    Collator that handles variable numbers of candidates for dual ranking tasks.
    It flattens the [Batch, Candidates, Tokens] structure and pads to the maximum
    number of candidates in the current batch.
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 128
    pad_to_multiple_of: Optional[int] = 8
    shuffle_candidates: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch.update(self._process_head(features, "occ", "occ_labels"))
        batch.update(self._process_head(features, "sec", "sector_labels"))
        return batch

    def _process_head(self, features, prefix, label_key):
        input_key = f"{prefix}_input_ids"
        # Return empty dict if the task data is missing from the features
        if not features or input_key not in features[0]: 
            return {}

        flat_input_ids, flat_att_masks, flat_type_ids = [], [], []
        choice_masks, labels = [], []
        
        max_k = max(len(f[input_key]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        for f in features:
            candidates = f[input_key]
            att = f.get(f"{prefix}_attention_mask", [None]*len(candidates))
            types = f.get(f"{prefix}_token_type_ids", [None]*len(candidates))
            lbl = f.get(label_key)

            if self.shuffle_candidates and lbl is not None:
                idxs = list(range(len(candidates)))
                random.shuffle(idxs)
                candidates = [candidates[i] for i in idxs]
                att = [att[i] for i in idxs]
                types = [types[i] for i in idxs]
                lbl = idxs.index(lbl)

            # Pad the number of candidates to match the batch max (max_k)
            k_curr = len(candidates)
            num_pad = max_k - k_curr
            
            flat_input_ids.extend(candidates + [[pad_id]] * num_pad)
            flat_att_masks.extend(att + [None] * num_pad)
            flat_type_ids.extend(types + [None] * num_pad)
            
            # Mask tracks which candidates are real vs padding
            choice_masks.append([True] * k_curr + [False] * num_pad)
            if lbl is not None: 
                labels.append(lbl)

        # Use tokenizer to pad the sequence dimension (T)
        batch_enc = self.tokenizer.pad(
            {"input_ids": flat_input_ids, "attention_mask": flat_att_masks, "token_type_ids": flat_type_ids},
            padding=True, 
            max_length=self.max_length, 
            pad_to_multiple_of=self.pad_to_multiple_of, 
            return_tensors="pt"
        )

        B, K = len(features), max_k
        def reshape(t): return t.view(B, K, -1)

        out = {
            f"{prefix}_input_ids": reshape(batch_enc["input_ids"]),
            f"{prefix}_attention_mask": reshape(batch_enc["attention_mask"]),
            f"{prefix}_choice_mask": torch.tensor(choice_masks, dtype=torch.bool),
        }
        
        if "token_type_ids" in batch_enc:
            out[f"{prefix}_token_type_ids"] = reshape(batch_enc["token_type_ids"])
            
        if labels:
            out[label_key] = torch.tensor(labels, dtype=torch.long)
            
        return out
