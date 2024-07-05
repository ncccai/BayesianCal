import torch
import torch.nn as nn
from torch import Tensor

class ResidualConvBlock(nn.Module):
    """Implements residual conv function.
	Args:
		channels (int): Number of channels in the input image.
	"""

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.PReLU(),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)
        out = torch.add(out, identity)

        return out

class BayesianCal(nn.Module):
    def __init__(self, in_channels=3, out_channels=3) -> None:
        super(BayesianCal, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels, 64,
                kernel_size=9, stride=1, padding=4
            ),
            nn.PReLU(),
        )

        # Features trunk blocks.
        trunk = []
        for _ in range(16):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                64, 64,
                kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm1d(64),
        )

        self.conv_block3_mu = nn.Conv1d(
            64, out_channels=out_channels,
            kernel_size=9, stride=1, padding=4
        )
        self.conv_block3_alpha = nn.Sequential(
            nn.Conv1d(
                64, 64,
                kernel_size=9, stride=1, padding=4
            ),
            nn.PReLU(),
            nn.Conv1d(
                64, 64,
                kernel_size=9, stride=1, padding=4
            ),
            nn.PReLU(),
            nn.Conv1d(
                64, 1,
                kernel_size=9, stride=1, padding=4
            ),
            nn.ReLU(),
        )
        self.conv_block3_beta = nn.Sequential(
            nn.Conv1d(
                64, 64,
                kernel_size=9, stride=1, padding=4
            ),
            nn.PReLU(),
            nn.Conv1d(
                64, 64,
                kernel_size=9, stride=1, padding=4
            ),
            nn.PReLU(),
            nn.Conv1d(
                64, 1,
                kernel_size=9, stride=1, padding=4
            ),
            nn.ReLU(),
        )

        # Initialize neural network weights.
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(2)
        return self._forward_impl(x)

    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = out1 + out2
        out_mu = self.conv_block3_mu(out)
        out_alpha = self.conv_block3_alpha(out)
        out_beta = self.conv_block3_beta(out)
        return out_mu, out_alpha, out_beta

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
