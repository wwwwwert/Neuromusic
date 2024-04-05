from typing import List

import torch
from torch import Tensor

from scripts.base.base_metric import BaseMetric
from scripts.metric.utils import calc_accuracy_score


class ArgmaxAccuracyMetric(BaseMetric):
    def __call__(self, logits: Tensor, target_ids: Tensor, sequence_length: List, **kwargs):
        predictions = torch.argmax(logits.cpu(), dim=-1).numpy()
        target_ids = target_ids.cpu().detach().clone()
        accuracies = []
        for pred, length, target in zip(predictions, sequence_length, target_ids):
            if length == 0:
                continue
            pred = pred[:length]
            target = target[:length]
            accuracies.append(calc_accuracy_score(target, pred))
        return sum(accuracies) / len(accuracies)
