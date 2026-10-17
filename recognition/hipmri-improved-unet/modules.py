import torch
import torch.nn as nn


class ContextModule(nn.Module):
    """
    2D adaptation of the pre-activation residual context module
    described by Isensee et al. (2018).

    Input and output feature channels remain the same.
    """

    def __init__(self, channels, dropout_p=0.3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.norm1 = nn.InstanceNorm2d(channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.dropout = nn.Dropout2d(p=dropout_p)

        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.norm2 = nn.InstanceNorm2d(channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(self.act1(self.norm1(x)))
        out = self.dropout(out)
        out = self.conv2(self.act2(self.norm2(out)))

        return out + residual


#level 1 testing- testing the ContextModule
if __name__ == "__main__":
    model = ContextModule(channels=32)

    x = torch.randn(1, 32, 256, 144)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
