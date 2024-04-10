from typing import List

from scripts.base.base_metric import BaseMetric


class MeanLengthMetric(BaseMetric):
    def __call__(self, sequence_length: List, **kwargs):
        return sum(sequence_length) / len(sequence_length)
