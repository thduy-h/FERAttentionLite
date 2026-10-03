import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ChannelAttention(nn.Module):
    """
    Channel attention used by FERAttentionLite.

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.mlp(self.pool(x))
        return x * weights


class SpatialAttention(nn.Module):
    """
    Spatial attention used by FERAttentionLite.

    Computes channel-wise average and maximum maps,
    concatenates them, then learns one spatial attention map.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)

        weights = self.attn(
            torch.cat([avg_map, max_map], dim=1)
        )

        return x * weights


class FERAttentionLite(nn.Module):
    """
    Exact FERAttentionLite architecture used in Kaggle Run 2 / Run 3.

    Input:
        grayscale FER-2013 face tensor [B, 1, 48, 48]

    Output:
        logits [B, 7]

    IMPORTANT:
    - Dropout values here match the tuned Run 2 / Run 3 model.
    - Do not change layer names/order if loading the existing best.pt.
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.block1 = nn.Sequential(
            ConvBNReLU(1, 32),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
        )

        self.block2 = nn.Sequential(
            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )

        self.block3_pre = nn.Sequential(
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
        )

        self.channel_attention = ChannelAttention(
            channels=128,
            reduction=8,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=7,
        )

        self.block3_post = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
        )

        self.block4 = nn.Sequential(
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            nn.Dropout2d(0.25),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)

        x = self.block3_pre(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.block3_post(x)

        x = self.block4(x)
        x = self.gap(x)

        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":
    model = FERAttentionLite(num_classes=7)
    model.eval()

    x = torch.randn(1, 1, 48, 48)

    with torch.inference_mode():
        y = model(x)

    print(model)
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Parameters  :", f"{count_parameters(model):,}")

    assert y.shape == (1, 7)
    print("Smoke test: PASS")
