"""CNN classifiers with attention pooling over time."""
import torch
from torch import nn

from .features import LogMel


class AttentionPool(nn.Module):
    """PSLA/PANNs-style pooling: per-frame class logits weighted by a softmax attention over time."""

    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.cla = nn.Conv1d(in_ch, n_classes, kernel_size=1)
        self.att = nn.Conv1d(in_ch, n_classes, kernel_size=1)

    def forward(self, x):  # [B, C, T]
        att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        return (self.cla(x) * att).sum(dim=-1)


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    """VGG-style baseline (PANNs CNN10-like), trained from scratch."""

    def __init__(self, n_classes, channels=(64, 128, 256, 512)):
        super().__init__()
        layers, cin = [], 1
        for c in channels:
            layers += [_conv_block(cin, c), nn.AvgPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.3)
        self.head = AttentionPool(cin, n_classes)

    def forward(self, x):  # [B, 1, F, T]
        x = self.features(x).mean(dim=2)  # average over frequency -> [B, C, T']
        return self.head(self.dropout(x))


class TimmCNN(nn.Module):
    """Any timm CNN (default EfficientNet-B0) on 1-channel spectrograms."""

    def __init__(self, name, n_classes, pretrained):
        super().__init__()
        import timm

        self.backbone = timm.create_model(name, pretrained=pretrained, in_chans=1, num_classes=0, global_pool="")
        self.dropout = nn.Dropout(0.3)
        self.head = AttentionPool(self.backbone.num_features, n_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x).mean(dim=2)
        return self.head(self.dropout(x))


class AudioModel(nn.Module):
    def __init__(self, frontend, net):
        super().__init__()
        self.frontend = frontend
        self.net = net

    def forward(self, wav, spec_transform=None):
        x = self.frontend(wav)
        if spec_transform is not None:
            x = spec_transform(x)
        return self.net(x)


def build_model(cfg, n_classes, pretrained=None):
    pretrained = cfg.pretrained if pretrained is None else pretrained
    if cfg.model == "simple_cnn":
        net = SimpleCNN(n_classes)
    else:
        net = TimmCNN(cfg.model, n_classes, pretrained)
    return AudioModel(LogMel.from_config(cfg), net)
