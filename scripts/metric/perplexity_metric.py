from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from scripts.base.base_metric import BaseMetric
from scripts.metric.utils import calc_perplexity


class PerplexityMetric(BaseMetric):
    def __call__(self, logits: Tensor, target_ids: Tensor, sequence_length: List, **kwargs):
        perplexities = []
        for item_logits, length, target in zip(logits, sequence_length, target_ids):
            if length == 0:
                continue
            item_logits = item_logits[:length]
            target = target[:length]
            perplexities.append(calc_perplexity(target, item_logits))
        return sum(perplexities) / len(perplexities)
