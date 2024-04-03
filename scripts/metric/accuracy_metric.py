import torch
from torch import Tensor

from scripts.base.base_metric import BaseMetric
from scripts.metric.utils import calc_accuracy_score
from typing import List


class ArgmaxAccuracyMetric(BaseMetric):
    def __call__(self, logits: Tensor, sequence_length: List, target_sequences: List, **kwargs):
        predictions = torch.argmax(logits.cpu(), dim=-1).numpy()
        accuracies = []
        for pred, length, target_sequence in zip(predictions, sequence_length, target_sequences):
            if length == 0:
                continue
            pred = pred[:length]
            target_sequences = target_sequences[:length]
            accuracies.append(calc_accuracy_score(target_sequence, pred))
        return sum(accuracies) / len(accuracies)
