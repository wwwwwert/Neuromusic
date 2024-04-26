from miditok.midi_tokenizer import MIDITokenizer
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM

from scripts.base import BaseModel


class Llama(BaseModel):
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

        self.config = LlamaConfig(
            vocab_size=n_class,
            hidden_size=d_model,
            intermediate_size=dim_feedforward,
            num_hidden_layers=n_layers,
            num_attention_heads=num_heads,
            pad_token_id=tokenizer["PAD_None"],
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"]
        )

        self.model = LlamaForCausalLM(self.config)

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
