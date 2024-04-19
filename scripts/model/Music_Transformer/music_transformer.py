from typing import Optional

from torch import Tensor, nn
from torch.nn.modules.normalization import LayerNorm

from scripts.base import BaseModel

from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderLayerRPR, TransformerEncoderRPR
from miditok.midi_tokenizer import MIDITokenizer



class MusicTransformer(BaseModel):
    """The implementation of Music Transformer.

    Music Transformer reproduction from https://arxiv.org/abs/1809.04281.
    Arguments allow for tweaking the transformer architecture
    (https://arxiv.org/abs/1706.03762) and the rpr argument toggles
    Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class
    with DummyDecoder to make a decoder-only transformer architecture.

    For RPR support, there is modified Pytorch 1.2.0 code in model/rpr.py.
    """

    def __init__(
        self,
        tokenizer: MIDITokenizer,
        input_length: int,
        n_layers: int=6,
        num_heads: int=8,
        d_model: int=512,
        dim_feedforward: int=1024,
        dropout: float=0.1,
        rpr: bool=True,
    ) -> None:
        """Inits MusicTransformer.

        Default parameters are taken from section 4.2 of the original article:
        https://arxiv.org/abs/1809.04281

        Args:
            n_out (int): vocab size, number of probabilities to return
            n_seq (int): length of input sequence
            n_layers (int): A number of layers in the encoder.
            num_heads (int): A number of heads used in Multi-Head attention.
            d_model (int): A token embedding size.
            dim_feedforward (int): A dimension of the feedforward network model
                used in nn.Transformer.
            dropout (float): A dropout value in Positional Encoding and in
                encoder layers.
            rpr (bool): A boolean value indicating whether to use Relative
                Positional Encoding or not.
        """
        n_class = len(tokenizer)
        pad_id = tokenizer["PAD_None"]
        super().__init__(n_class, input_length)

        self.dummy = DummyDecoder()
        self.nlayers = n_layers
        self.nhead = num_heads
        self.d_model = d_model
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.max_seq = input_length

        # Input embedding
        self.embedding = nn.Embedding(
            n_class,
            self.d_model,
            padding_idx=pad_id,
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.d_model,
            dropout=self.dropout,
            max_len=self.max_seq,
        )

        # Define encoder as None for Base Transformer
        encoder = None

        # else define encoder as TransformerEncoderRPR for RPR Transformer
        if rpr:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(
                self.d_model,
                self.nhead,
                dim_feedforward=self.d_ff,
                p_dropout=self.dropout,
                er_len=self.max_seq,
            )
            encoder = TransformerEncoderRPR(
                encoder_layer,
                self.nlayers,
                norm=encoder_norm,
            )

        # To make a decoder-only transformer we need to use masked encoder
        # layers and DummyDecoder to essentially just return the encoder output
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.nlayers,
            num_decoder_layers=0,
            dropout=self.dropout,
            dim_feedforward=self.d_ff,
            custom_decoder=self.dummy,
            custom_encoder=encoder
        )

        self.Wout = nn.Linear(self.d_model, n_class)

    def forward(
        self,
        input_ids: Tensor,
        padding_mask: bool = True,
        **batch
    ) -> Tensor:
        """Takes an input sequence and outputs predictions via seq2seq method.

        A prediction at one index is the "next" prediction given all information
        seen previously.

        Args:
            x (Tensor): A tensor of tokenized input compositions of
                dimension (batch_size, self.max_seq).

        Returns:
            A tensor of dimension (batch_size, self.max_seq, n_out), where
            in the position [i, j, :] are the logits of the distribution of the
            `j+1`-th token of the `i`-th composition from the batch.
        """
        mask = self.transformer.generate_square_subsequent_mask(
            input_ids.shape[1]
        ).to(input_ids.device)
        x = self.embedding(input_ids)

        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        padding_mask = (padding_mask != 1)
        x_out = self.transformer(src=x, tgt=x, src_mask=mask) #, src_key_padding_mask=padding_mask)
        # masking somehow breaks gradients (nans in gradients)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1, 0, 2)

        y = self.Wout(x_out)

        del mask  # noqa: WPS420

        # They are trained to predict the next note in sequence
        # we don't need the last one
        return {"logits": y}


class DummyDecoder(nn.Module):
    """A dummy decoder that returns its input.

    Used to make the Pytorch transformer into a decoder-only architecture
    (stacked encoders with dummy decoder fits the bill).
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        **args
    ) -> Tensor:
        """Returns the input (memory).

        Args:
            tgt (Tensor): A sequence to the decoder.
            memory (Tensor): A sequence from the last layer of the encoder.
            tgt_mask (Optional[Tensor]): A mask for the tgt sequence.
            memory_mask (Optional[Tensor]): A mask for the memory sequence.
            tgt_key_padding_mask (Optional[Tensor]): A mask for the tgt keys per
                batch.
            memory_key_padding_mask (Optional[Tensor]): A mask for the memory
                keys per batch.

        Returns:
            The `memory` tensor from input.
        """
        return memory
