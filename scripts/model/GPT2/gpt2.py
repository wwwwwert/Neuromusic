from typing import Optional

from miditok.midi_tokenizer import MIDITokenizer
from torch import Tensor, nn
from torch.nn.modules.normalization import LayerNorm
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.base import BaseModel


class GPT2(BaseModel):
    def __init__(
        self,
        tokenizer: MIDITokenizer,
        input_length: int,
        n_layers: int=6,
        num_heads: int=8,
        d_model: int=768,
        dim_feedforward: int=1024,
    ) -> None:
        n_class = len(tokenizer)
        super().__init__(n_class, input_length)

        self.config = GPT2Config(
            vocab_size=n_class,
            n_positions=input_length,
            n_embd=d_model,
            n_layer=n_layers,
            n_head=num_heads,
            n_inner=dim_feedforward,
            pad_token_id=tokenizer["PAD_None"],
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"]
        )

        self.model = GPT2LMHeadModel(self.config)

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Tensor,
        **batch
    ) -> Tensor:
        output = self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            return_dict=True
        )
        logits = output['logits']
        return {"logits": logits}
