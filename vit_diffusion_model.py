import torch
from torch import nn
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel


class DiTNoisePredictor(nn.Module):
    """Diffusion Transformer noise predictor (DiT: Peebles & Xie, "Scalable Diffusion
    Models with Transformers", ICCV 2023, arXiv:2212.09748).

    Thin wrapper around diffusers' DiTTransformer2DModel that exposes the same
    forward(x, t) interface as chuchichaestli.models.unet.unet.UNet, so it can
    be dropped into the existing DDPM training/sampling loop unmodified.

    DiT is class-conditional by design (AdaLN-Zero mixes a class embedding into
    every block). For unconditional generation we register a single dummy class
    (num_embeds_ada_norm=1) and always pass class label 0.
    """

    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        patch_size: int = 8,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        attention_head_dim: int = 64,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
    ) -> None:
        super().__init__()
        self.dit = DiTTransformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            sample_size=sample_size,
            patch_size=patch_size,
            activation_fn=activation_fn,
            num_embeds_ada_norm=1,
        )

    def forward(
        self, x: torch.Tensor, t: int | float | torch.Tensor | None = None
    ) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.long, device=x.device)
        t = t.expand(x.shape[0]).to(device=x.device, dtype=torch.long)
        class_labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.dit(
            hidden_states=x,
            timestep=t,
            class_labels=class_labels,
            return_dict=False,
        )[0]
