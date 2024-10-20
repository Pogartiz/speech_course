from typing import Optional, Union

import torch
import torch.nn
from omegaconf import DictConfig

from src.submodules.subsampling import StackingSubsampling
from src.submodules.positional_encoding import PositionalEncoding


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_linear_after_conv: bool = False,
    ):
        """
        Convolutional Spatial Gating Unit (https://arxiv.org/pdf/2207.02971)
        Args:
            size: int - Input embedding dim
            kernel_size: int - Kernel size in DepthWise Conv
            dropout: float - Dropout rate
            use_linear_after_conv: bool - Whether to use linear layer after convolution
        """

        super().__init__()
        self.norm = torch.nn.LayerNorm(size // 2)

        self.conv = torch.nn.Conv1d(
            size // 2, size // 2, kernel_size, groups=size // 2, padding=(kernel_size - 1) // 2
        )

        if use_linear_after_conv:
            self.linear = torch.nn.Linear(size // 2, size // 2)
        else:
            self.linear = None

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Inputs:
            x: B x T x C
        Outputs:
            out: B x T x C
        """
        x_r, x_g = x.chunk(2, dim=-1)

        x_g = self.norm(x_g).transpose(1, 2)
        x_g = self.conv(x_g).transpose(1, 2)

        if self.linear is not None:
            x_g = self.linear(x_g)

        out = self.dropout(x_r * x_g)
        return out


class ConvolutionalGatingMLP(torch.nn.Module):
    def __init__(
        self,
        size: int,
        kernel_size: int,
        expansion_factor: int = 6,
        dropout: float = 0.0,
        use_linear_after_conv: bool = False,
    ):
        """
        Convolutional Gating MLP (https://arxiv.org/pdf/2207.02971)
        Args:
            size: int - Input embedding dim
            kernel_size: int - Kernel size for DepthWise Conv in ConvolutionalSpatialGatingUnit
            expansion_factor: int - Dim expansion factor for ConvolutionalSpatialGatingUnit
            dropout: float - Dropout rate
            use_linear_after_conv: bool - Whether to use linear layer after convolution
        """
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, size * expansion_factor),
            torch.nn.GELU()
        )

        self.csgu = ConvolutionalSpatialGatingUnit(
            size=size * expansion_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            use_linear_after_conv=use_linear_after_conv
        )

        self.channel_proj2 = torch.nn.Sequential(torch.nn.Linear(size * expansion_factor // 2, size))


    def forward(self, features: torch.Tensor):
        """
        Inputs:
            features: B x T x C
        Outputs:
            out: B x T x C
        """
        features = self.channel_proj1(features)
        features = self.csgu(features)
        features = self.channel_proj2(features)

        return features


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: torch.nn.Module = torch.nn.SiLU(),
    ):
        """
        Standard FeedForward layer from Transformer block,
        consisting of a compression and decompression projection
        with an activation function.
        Args:
            input_dim: int - Input embedding dim
            hidden_dim: int - Hidden dim
            dropout: float - Dropout rate
            activation: torch.nn.Module - Activation function
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, input_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation

    def forward(self, features: torch.Tensor):
        """
        Inputs:
            features: B x T x C
        Outputs:
            out: B x T x C
        """
        features = self.linear1(features)
        features = self.activation(features)
        features = self.dropout(features)
        features = self.linear2(features)
        features = self.dropout(features)

        return features


class EBranchformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        size: int,
        attn_config: Union[DictConfig, dict],
        cgmlp_config: Union[DictConfig, dict],
        ffn_expansion_factor: int = 4,
        dropout: float = 0.0,
        merge_conv_kernel: int = 3,
    ):
        """
        E-Branchformer Layer (https://arxiv.org/pdf/2210.00077)
        Args:
            size: int - Embedding dim
            attn_config: DictConfig or dict - Config for MultiheadAttention
            cgmlp_config: DictConfig or dict - Config for ConvolutionalGatingMLP
            ffn_expansion_factor: int - Expansion factor for FeedForward
            dropout: float - Dropout rate
            merge_conv_kernel: int - Kernel size for merging module
        """
        super().__init__()

        # attn
        self.attn = torch.nn.MultiheadAttention(**attn_config)

        # cgmlp
        self.cgmlp = ConvolutionalGatingMLP(**cgmlp_config)

        # feedforwrd
        self.feed_forward1 = FeedForward(input_dim=size, hidden_dim=size * ffn_expansion_factor, dropout=dropout)
        self.feed_forward2 = FeedForward(input_dim=size, hidden_dim=size * ffn_expansion_factor, dropout=dropout)

        # norm
        self.norm_ffn1 = torch.nn.LayerNorm(size)
        self.norm_ffn2 = torch.nn.LayerNorm(size)
        self.norm_mha = torch.nn.LayerNorm(size)
        self.norm_mlp = torch.nn.LayerNorm(size)
        self.norm_final = torch.nn.LayerNorm(size)

        self.dropout = torch.nn.Dropout(dropout)

        # depthwise
        self.depthwise_conv_fusion = torch.nn.Conv1d(
            size * 2, size * 2, kernel_size=merge_conv_kernel, stride=1,
            padding=(merge_conv_kernel - 1) // 2, groups=size * 2
        )
        self.merge_proj = torch.nn.Linear(size * 2, size)

    def forward(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
    ):
        """
        Inputs:
            features: B x T x C
            features_length: B
            pos_emb: B x T x C - Optional
        Outputs:
            out: B x T x C
        """
        device = features.device

        # ffn1
        residual = features
        x = self.norm_ffn1(features)
        x = self.feed_forward1(x)
        x = residual + 0.5 * x

        # attn        
        x_gl = self.norm_mha(x)
        if pos_emb is not None:
            x_gl += pos_emb
        key_padding_mask = (torch.arange(x_gl.size(1), device=device).unsqueeze(0).expand(features_length.size(0), x_gl.size(1)) >= features_length.unsqueeze(1))
        key_padding_mask = key_padding_mask.transpose(0, 1)

        attn_output, _ = self.attn(x_gl, x_gl, x_gl, key_padding_mask=key_padding_mask)
        x_global = self.dropout(attn_output)

        # cgmlp
        residual = x
        x = self.norm_mlp(x)
        x_local = self.cgmlp(x)

        # Merge global and local
        x_concat = torch.cat([x_global, x_local], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x_merged = self.merge_proj(x_concat + x_tmp)

        # Residual Connection after Merging
        x = residual + self.dropout(x_merged)

        # Second FeedForward with residual
        residual = x
        x = self.norm_ffn2(x)
        x = self.feed_forward2(x)
        x = residual + 0.5 * x

        x = self.norm_final(x)

        return x




class EBranchformerEncoder(torch.nn.Module):
    def __init__(
        self,
        subsampling_stride: int,
        features_num: int,
        d_model: int,
        layers_num: int,
        attn_config: Union[DictConfig, dict],
        cgmlp_config: Union[DictConfig, dict],
        ffn_expansion_factor: int = 2,
        dropout: float = 0.0,
        merge_conv_kernel: int = 3,
    ):
        super().__init__()
        self.subsampling = StackingSubsampling(
            stride=subsampling_stride, feat_in=features_num, feat_out=d_model
        )
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        self.layers = torch.nn.ModuleList()
        for _ in range(layers_num):
            layer = EBranchformerEncoderLayer(
                size=d_model,
                attn_config=attn_config,
                cgmlp_config=cgmlp_config,
                ffn_expansion_factor=ffn_expansion_factor,
                dropout=dropout,
                merge_conv_kernel=merge_conv_kernel,
            )
            self.layers.append(layer)

    def forward(self, features: torch.Tensor, features_length: torch.Tensor):
        features = features.transpose(1, 2)  # B x D x T -> B x T x D
        features, features_length = self.subsampling(features, features_length)
        features = self.pos_embedding(features)
        for layer in self.layers:
            features = layer(features, features_length)

        return features, features_length
