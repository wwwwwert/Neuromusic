from typing import Any, Optional, Tuple, Union
from warnings import warn

import torch
from torch.nn import Module
from torch.nn.functional import dropout, linear, pad, relu, softmax
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.parameter import Parameter


class TransformerEncoderRPR(Module):
    """The implementation of TransformerEncoder from pytorch.

    For Relative Position Representation support:
    https://arxiv.org/abs/1803.02155

    No modification.
    Copied from to ensure continued compatibility with other edits:
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/transformer.html#TransformerEncoder
    """

    def __init__(
        self,
        encoder_layer: Module,
        num_layers: int,
        norm: Optional[LayerNorm] = None,
    ) -> None:
        """Inits Transformer Encoder with custom Encoder Layer.

        Args:
            encoder_layer (Module): A nn.Module that defines the custom encoder
                layer
            num_layers (int): A number of layers in the encoder.
            norm (Optional[LayerNorm]): An optional layer normalization
                component.
        """
        super().__init__()
        self.layers = torch.nn.modules.transformer._get_clones(  # noqa: WPS437
            encoder_layer,
            num_layers,
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        **args
    ) -> torch.Tensor:
        """Applies a sequence of encoder layers.

        Args:
            src (Tensor): An input sequence tensor of dimension (max_seq,
                batch_size, d_model).
            mask (Optional[Tensor]): An optional mask for the `src` sequence.
            src_key_padding_mask (Optional[Tensor]): An optional mask for the
                src keys per batch.

        Returns:
            A tensor of dimension (max_seq, batch_size, d_model).
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayerRPR(Module):
    """The implementation of Transformer encoder layer with RPR support.

    For Relative Position Representation support:
    https://arxiv.org/abs/1803.02155
    The standard implementation of the encoder layer lies here:
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer

    Modification to create and call custom MultiheadAttentionRPR.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        p_dropout: float = 0.1,
        er_len: Optional[int] = None,
    ) -> None:
        """Inits encoder layer with Multi-Head Attention with RPR.

        Args:
            d_model (int): A token embedding size.
            nhead (int): A number of heads used in the Multi-Head Attention with
                RPR.
            dim_feedforward (int):  A dimension of the feedforward network model
                used after using the Multi-Head Attention with RPR.
            p_dropout (float): A dropout value used in the Multi-Head Attention
                with RPR and in the feedforward network.
            er_len (Optional[int]): A maximum length of a tokenized composition.
        """
        super().__init__()
        self.self_attn = MultiheadAttentionRPR(
            d_model,
            nhead,
            p_dropout=p_dropout,
            er_len=er_len,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(p=p_dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(p=p_dropout)
        self.dropout2 = Dropout(p=p_dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies Multi-Head Attention with RPR and feedforward network model.

        Args:
            src (Tensor): An input sequence tensor of dimension (max_seq,
                batch_size, d_model).
            src_mask (Optional[Tensor]): An optional mask for the `src`
                sequence.
            src_key_padding_mask (Optional[Tensor]): An optional mask for the
                src keys per batch.

        Returns:
            A tensor of dimension (max_seq, batch_size, d_model).
        """
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src  # noqa: WPS331


class MultiheadAttentionRPR(Module):
    """The implementation of Multi-Head Attention with RPR support.

    For Relative Position Representation support:
    https://arxiv.org/abs/1803.02155
    The standard implementation of the Multi-Head Attention lies here:
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/activation.html#MultiheadAttention

    Modification to add RPR embedding `Er` and call custom
    `multi_head_attention_forward_rpr`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        p_dropout: float = 0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        er_len: Optional[int] = None,
    ) -> None:
        """Inits Multi-Head Attention with RRP support.

        Args:
            embed_dim (int): A token embedding size.
            num_heads (int): A number of heads.
            p_dropout (float): A dropout value.
            bias (bool): A boolean value indicating whether to add bias as
                module parameter or not.
            add_bias_kv (bool): A boolean value indicating whether to add bias
                to the key and value sequences at `dim` = 0.
            add_zero_attn (bool): A boolean value indicating whether to add a
                new batch of zeros to the key and value sequences at `dim` = 1.
            kdim (Optional[int]): A total number of features in key.
            vdim (Optional[int]): A total number of features in value.
            er_len (Optional[int]): A maximum length of a tokenized composition.

        Raises:
            AssertionError: embed_dim must be divisible by num_heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim == self.vdim

        self.num_heads = num_heads
        self.dropout = p_dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise AssertionError(f'embed_dim (got {embed_dim}) must be divisible by num_heads (got {num_heads})')  # noqa: E501

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # Adding RPR embedding matrix
        if er_len is not None:
            self.Er = Parameter(
                torch.rand((er_len, self.head_dim), dtype=torch.float32),
            )
        else:
            self.Er = None

        self._reset_parameters()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Applies Multi-Head Attention with RPR.

        The whole logic of using this attention mechanism is written in the
        function `multi_head_attention_forward_rpr`.

        Args:
            query (Tensor): A query embeddings of dimension
                `(max_sequence, batch_size, embedding_dim)`.
            key (Tensor): A key embeddings of dimension
                `(max_sequence, batch_size, embedding_dim)`.
            value (Tensor): A value embeddings of dimension
                `(max_sequence, batch_size, embedding_dim)`.
            key_padding_mask (Optional[Tensor]): An optional mask of dimension
                `(batch_size, max_sequence)` indicating which elements within
                key to ignore for the purpose of attention.
            need_weights (bool): A boolean value indicating whether to return
                attention weights in addition to attention outputs.
            attn_mask (Optional[Tensor]): A 2D or 3D optional mask preventing
                attention to certain positions.

        Returns:
            attention_output (Tensor): An attention output tensor of dimension
                `(max_sequence, batch_size, embedding_dim)`.
            attention_weights (Optional[Tensor]): An optional output, an
                attention weights of dimension `(batch_size, embedding_dim,
                max_sequence)`. Only returned when `need_weights` is True.

        Warnings:
            UserWarning: A warning occurred if old version of Multi-Head
                Attention had used.
        """
        multi_head_attention_forward_rpr_kwargs = {
            'training': self.training,
            'key_padding_mask': key_padding_mask,
            'need_weights': need_weights,
            'attn_mask': attn_mask,
            'rpr_mat': self.Er,
        }

        if hasattr(self, '_qkv_same_embed_dim'):
            if not self._qkv_same_embed_dim:
                multi_head_attention_forward_rpr_kwargs.update({
                    'use_separate_proj_weight': True,
                    'q_proj_weight': self.q_proj_weight,
                    'k_proj_weight': self.k_proj_weight,
                    'v_proj_weight': self.v_proj_weight,
                })
        else:
            warn(
                'A new version of Multi-Head Attention module has been implemented. Please re-train your model with the new module.',  # noqa: E501
                UserWarning,
            )

        return multi_head_attention_forward_rpr(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            **multi_head_attention_forward_rpr_kwargs,
        )

    def _reset_parameters(self) -> None:
        """The initialization of the Multi-Head Attention inner layers."""
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0)
            constant_(self.out_proj.bias, 0)

        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)


# this is a really heavy function, we will disable complexity checks
def multi_head_attention_forward_rpr(  # noqa: WPS2**
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: torch.Tensor,
    in_proj_bias: Optional[torch.Tensor],
    bias_k: Optional[torch.Tensor],
    bias_v: Optional[torch.Tensor],
    add_zero_attn: bool,
    p_dropout: float,
    out_proj_weight: torch.Tensor,
    out_proj_bias: Optional[torch.Tensor],
    training: bool = True,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[torch.Tensor] = None,
    k_proj_weight: Optional[torch.Tensor] = None,
    v_proj_weight: Optional[torch.Tensor] = None,
    static_k: Optional[torch.Tensor] = None,
    static_v: Optional[torch.Tensor] = None,
    rpr_mat: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Performs Multi-Head Attention with RPR.

    For Relative Position Representation support:
    https://arxiv.org/abs/1803.02155
    The standard implementation of the Multi-Head Attention lies here:
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/activation.html#MultiheadAttention

    Modification to take RPR embedding matrix and perform skew optimized RPR:
    https://arxiv.org/abs/1809.04281

    Args:
        query (Tensor): A query embeddings of dimension (max_sequence,
            batch_size, embedding_dim).
        key (Tensor): A key embeddings of dimension (max_sequence, batch_size,
            embedding_dim).
        value (Tensor): A value embeddings of dimension (max_sequence,
            batch_size, embedding_dim).
        embed_dim_to_check (int): A reference token embedding size.
        num_heads (int): A number of heads.
        in_proj_weight (Tensor): A learnable matrix of linear transformation
            into Query, Key, Value tensors.
        in_proj_bias (Optional[Tensor]): An optional learnable bias of linear
            transformation into Query, Key, Value tensors.
        bias_k (Optional[Tensor]): An optional bias to the key sequence at `dim`
            = 0.
        bias_v (Optional[Tensor]): An optional bias to the value sequence at
            `dim` = 0.
        add_zero_attn (bool): A boolean value indicating whether to add a
            new batch of zeros to the key and value sequences at `dim` = 1.
        p_dropout (float): A dropout value.
        out_proj_weight (Tensor): A learnable matrix of linear transformation
            after applying Self-Attention mechanism.
        out_proj_bias (Optional[Tensor]): A learnable matrix of linear
            transformation after applying Self-Attention mechanism.
        training (bool): A boolean value indicating whether to use dropout or
            not. Dropout is used during training and is disabled during
            inference.
        key_padding_mask (Optional[Tensor]): An optional mask of dimension
            `(batch_size, max_sequence)` indicating which elements within
        need_weights (bool): A boolean value indicating whether to return
            attention weights in addition to attention outputs.
        attn_mask (Optional[Tensor]): A 2D or 3D optional mask preventing
            attention to certain positions.
        use_separate_proj_weight (bool): A boolean value indicating whether to
            use separate matrices in the linear transformation into Query, Key,
            Value tensors.
        q_proj_weight (Optional[Tensor]): An optional learnable matrix of
            linear transformation into Query tensor. It is not None, if
            `use_separate_proj_weight` is True.
        k_proj_weight (Optional[Tensor]): An optional learnable matrix of
            linear transformation into Key tensor. It is not None, if
            `use_separate_proj_weight` is True.
        v_proj_weight (Optional[Tensor]): An optional learnable matrix of
            linear transformation into Value tensor. It is not None, if
            `use_separate_proj_weight` is True.
        static_k (Optional[Tensor]): An optional static tensor denoting Key
            tensor.
        static_v (Optional[Tensor]): An optional static tensor denoting Value
            tensor.
        rpr_mat (Optional[Tensor]): An optional tensor of relative positional
            representations of dimension `(max_sequence, embedding_size //
            num_heads)`.

    Returns:
        attention_output (Tensor): An attention output tensor of dimension
            `(max_sequence, batch_size, embedding_dim)`.
        attention_weights (Optional[Tensor]): An optional output, an
            attention weights of dimension `(batch_size, embedding_dim,
            max_sequence)`. Only returned when `need_weights` is True.

    Raises:
        AssertionError: There are a lot of checks that the dimensions of the
            tensors coincide.
    """
    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    _check_sizes_equality(embed_dim, embed_dim_to_check)
    _check_sizes_equality(key.size(), value.size())

    head_dim = embed_dim // num_heads

    if use_separate_proj_weight:
        q_proj_weight_non_opt = _unwrap_optional(q_proj_weight)
        _check_sizes_equality(q_proj_weight_non_opt.size(dim=0), embed_dim)
        _check_sizes_equality(
            q_proj_weight_non_opt.size(dim=-1),
            query.size(dim=-1),
        )

        k_proj_weight_non_opt = _unwrap_optional(k_proj_weight)
        _check_sizes_equality(k_proj_weight_non_opt.size(dim=0), embed_dim)
        _check_sizes_equality(
            k_proj_weight_non_opt.size(dim=-1),
            key.size(dim=-1),
        )

        v_proj_weight_non_opt = _unwrap_optional(v_proj_weight)
        _check_sizes_equality(v_proj_weight_non_opt.size(dim=0), embed_dim)
        _check_sizes_equality(
            v_proj_weight_non_opt.size(dim=-1),
            value.size(dim=-1),
        )

        if in_proj_bias is not None:
            q = linear(
                query,
                q_proj_weight_non_opt,
                bias=in_proj_bias[: embed_dim],
            )
            k = linear(
                key,
                k_proj_weight_non_opt,
                bias=in_proj_bias[embed_dim: embed_dim * 2],
            )
            v = linear(
                value,
                v_proj_weight_non_opt,
                bias=in_proj_bias[embed_dim * 2:],
            )
        else:
            q = linear(query, q_proj_weight_non_opt, bias=in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, bias=in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, bias=in_proj_bias)
    elif qkv_same:
        # self-attention
        q, k, v = linear(
            query,
            in_proj_weight,
            bias=in_proj_bias,
        ).chunk(3, dim=-1)
    else:
        # encoder-decoder attention
        q = _apply_linear(
            query,
            in_proj_weight,
            in_proj_bias=in_proj_bias,
            start=0,
            end=embed_dim,
        )

        if kv_same:
            if key is None:
                if value is not None:
                    raise AssertionError('key and value should be both None, but key is None and value is not')  # noqa: E501
                k = None
                v = None
            else:
                k, v = _apply_linear(
                    key,
                    in_proj_weight,
                    in_proj_bias=in_proj_bias,
                    start=embed_dim,
                    end=in_proj_weight.size(dim=0),
                ).chunk(2, dim=-1)
        else:
            k = _apply_linear(
                key,
                in_proj_weight,
                in_proj_bias=in_proj_bias,
                start=embed_dim,
                end=embed_dim * 2,
            )

            v = _apply_linear(
                value,
                in_proj_weight,
                in_proj_bias=in_proj_bias,
                start=embed_dim * 2,
                end=in_proj_weight.size(dim=0),
            )

    scaling = float(head_dim) ** -0.5
    q = q * scaling   # noqa: WPS350

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat((k, bias_k.repeat(1, bsz, 1)))
            v = torch.cat((v, bias_v.repeat(1, bsz, 1)))
            if attn_mask is not None:
                attn_mask = _pad_tensor_with_zeros(
                    attn_mask, (attn_mask.size(dim=0), 1), dim_to_pad=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = _pad_tensor_with_zeros(
                    key_padding_mask, (key_padding_mask.size(dim=0), 1),
                )
        elif static_k is not None:
            raise AssertionError('bias cannot be added to static key')
        elif static_v is not None:
            raise AssertionError('bias cannot be added to static value')
    elif bias_k is not None:
        raise AssertionError('bias_k and bias_v should be both None, but bias_v is None and bais_k is not')  # noqa: E501
    elif bias_v is not None:
        raise AssertionError('bias_k and bias_v should be both None, but bias_k is None and bais_v is not')  # noqa: E501

    # q's size is (tgt_len, bsz * num_heads, head_dim)
    size = (-1, bsz * num_heads, head_dim)  # noqa: WPS204
    q = q.contiguous().view(*size).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(*size).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(*size).transpose(0, 1)

    if static_k is not None:
        _check_sizes_equality(static_k.size(dim=0), bsz * num_heads)
        _check_sizes_equality(static_k.size(dim=2), head_dim)
        k = static_k

    if static_v is not None:
        _check_sizes_equality(static_v.size(dim=0), bsz * num_heads)
        _check_sizes_equality(static_v.size(dim=2), head_dim)
        v = static_v

    src_len = k.size(dim=1)

    if key_padding_mask is not None:
        _check_sizes_equality(key_padding_mask.size(dim=0), bsz)
        _check_sizes_equality(key_padding_mask.size(dim=1), src_len)

    if add_zero_attn:
        src_len += 1
        k_dim0, _, *k_dims = k.size()
        k = _pad_tensor_with_zeros(k, (k_dim0, 1, k_dims), dim_to_pad=1)

        v_dim0, _, *v_dims = v.size()
        v = _pad_tensor_with_zeros(v, (v_dim0, 1, v_dims), dim_to_pad=1)

        if attn_mask is not None:
            attn_mask = _pad_tensor_with_zeros(
                attn_mask, (attn_mask.size(dim=0), 1), dim_to_pad=1,
            )

        if key_padding_mask is not None:
            key_padding_mask = _pad_tensor_with_zeros(
                key_padding_mask, (key_padding_mask.size(dim=0), 1),
            )

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    _check_sizes_equality(
        attn_output_weights.size(),
        (bsz * num_heads, tgt_len, src_len),
    )

    # ADDITION OF RPR
    if rpr_mat is not None:
        rpr_mat = _get_valid_embedding(rpr_mat, q.size(dim=1))
        qe = torch.einsum('hld,md->hlm', q, rpr_mat)
        srel = _skew(qe)

        attn_output_weights += srel

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(
            (bsz, num_heads, tgt_len, src_len),
        )
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(
            (bsz * num_heads, tgt_len, src_len),
        )

    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(
        attn_output_weights,
        p=p_dropout,
        training=training,
    )

    attn_output = torch.bmm(attn_output_weights, v)
    _check_sizes_equality(
        attn_output.size(),
        (bsz * num_heads, tgt_len, head_dim),
    )
    attn_output = attn_output.transpose(0, 1).contiguous().view(
        (tgt_len, bsz, embed_dim),
    )
    attn_output = linear(attn_output, out_proj_weight, bias=out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(
            (bsz, num_heads, tgt_len, src_len),
        )
        return attn_output, attn_output_weights.sum(dim=1) / num_heads

    return attn_output, None


def _apply_linear(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: Optional[torch.Tensor] = None,
    start: int = 0,
    end: int = 0,
) -> torch.Tensor:
    """Applies linear transformation with given weights matrix and bias.

    The main difference from `torch.functional.linear` is getting needed slices
    from weights matrix and bias.

    Args:
        x (Tensor): An input tensor for applying linear transformation.
        in_proj_weight (Tensor): A weights' matrix.
        in_proj_bias (Optional[Tensor]): An optional bias.
        start (int): A beginning of the slice by `dim` = 0.
        end (int): An ending of the slice by `dim` = 0.

    Returns:
        A tensor after applying linear transformation.
    """
    weight = in_proj_weight[start: end, :]
    bias = in_proj_bias
    if bias is not None:
        bias = bias[start: end]
    return linear(x, weight, bias=bias)


def _check_sizes_equality(
    size0: Union[int, Tuple[int, ...]],
    size1: Union[int, Tuple[int, ...]],
) -> None:
    """Checks equality of given tensors' sizes or their separate dimensions.

    Args:
        size0 (Union[int, Tuple[int, ...]]): A size or dimension of the first
            tensor.
        size1 (Union[int, Tuple[int, ...]]): A size or dimension of the second
            tensor.

    Raises:
        AssertionError: If sizes or dimensions do not match.
    """
    if size0 != size1:
        raise AssertionError(
            f'size0 (got {size0}) should be equal to size1 (got {size1})',
        )


def _pad_tensor_with_zeros(
    x: torch.Tensor,
    zero_tensor_shape: Tuple[int, ...],
    dim_to_pad: int = 1,
) -> torch.Tensor:
    """Pads the tensor with zeros in the specified dimension.

    Args:
        x (Tensor):  An input tensor for padding.
        zero_tensor_shape (Tuple[int, ...]): A shape of zeros containing tensor.
        dim_to_pad (int): A dimension to pad.

    Returns:
        A padded by zeros input tensor.
    """
    zero_tensor = torch.zeros(
        zero_tensor_shape,
        dtype=x.dtype,
        device=x.device,
    )
    return torch.cat((x, zero_tensor), dim=dim_to_pad)


def _get_valid_embedding(er: torch.Tensor, len_q: int) -> torch.Tensor:
    """Gets valid embeddings based on max length of RPR attention.

    For more details see Section 3.4: https://arxiv.org/abs/1809.04281

    Args:
        er (Tensor): A tensor of relative positional representations of
            dimension `(max_sequence, embedding_size // num_heads)`.
        len_q (int): A dimension of Query tensor denoting maximum sequence
            length.

    Returns:
        A tensor with valid embeddings.
    """
    len_e = er.shape[0]
    start = max(0, len_e - len_q)
    return er[start:, :]


def _skew(qe: torch.Tensor) -> torch.Tensor:
    """Performs the skew optimized RPR computation.

    For more details see Section 3.4.1: https://arxiv.org/abs/1809.04281

    Args:
        qe (Tensor): An input tensor (in terms of the original paper $Q E^r$).

    Returns:
        A skewed input tensor.
    """
    sz = qe.size(dim=1)
    mask = torch.ones(sz, sz).to(qe.device)
    mask = (torch.triu(mask) == 1).float().flip(0)

    qe = mask * qe
    qe = pad(qe, (1, 0, 0, 0, 0, 0))

    qe = torch.reshape(qe, (-1, sz + 1, sz))

    return qe[:, 1:, :]  # srel in the notation of the article


def _unwrap_optional(x: Any) -> Any:
    """Unwraps Optional object.

    Args:
        x (Any): An input object.

    Returns:
        An input object if it is not None.

    Raises:
        AssertionError: occurs if input object is None.
    """
    if x is None:
        raise AssertionError('Unwrapping null optional')
    return x
