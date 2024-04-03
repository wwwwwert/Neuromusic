import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CELossWrapper(CrossEntropyLoss):
    def forward(self, logits, input_ids, **batch) -> Tensor:
        return super().forward(
            logits.permute(0, 2, 1),
            input_ids
        )
