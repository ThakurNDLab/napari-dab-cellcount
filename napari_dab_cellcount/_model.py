import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, Sequential

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        BatchNorm2d(out_channels),
        ReLU(inplace=True),
        Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        BatchNorm2d(out_channels),
        ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels, output_size=None):
    if output_size is None:
        return ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # Custom resize operation for non-standard dimensions
        return nn.Sequential(
            ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
        )

class ResNetUnet(nn.Module):
    def __init__(self, *, weights=None, out_channels=1):
        super().__init__()

        # Load pretrained ResNet-50
        pretrained_model = resnet50(weights=weights)

        # Encoder
        self.initial = Sequential(*list(pretrained_model.children())[:4]) # Initial layers
        self.encoder1 = pretrained_model.layer1 # First residual block
        self.encoder2 = pretrained_model.layer2 # Second residual block
        self.encoder3 = pretrained_model.layer3 # Third residual block
        self.encoder4 = pretrained_model.layer4 # Fourth residual block

        # Bottleneck
        self.bottleneck = double_conv(2048, 4096)

        # Decoder
        self.up_conv5 = up_conv(4096, 1024)
        self.conv5 = double_conv(2048 + 1024, 1024)
        self.up_conv6 = up_conv(1024, 512, output_size=(40, 40))  # Adjusted for 320x320 input
        self.conv6 = double_conv(1024 + 512, 512)
        self.up_conv7 = up_conv(512, 256, output_size=(80, 80))  # Adjusted for 320x320 input
        self.conv7 = double_conv(512 + 256, 256)
        self.up_conv8 = up_conv(256, 64, output_size=(160, 160))  # Adjusted for 320x320 input
        self.conv8 = double_conv(256 + 64, 64)
        self.final_conv = Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        initial = self.initial(x)
        enc1 = self.encoder1(initial)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        x = self.bottleneck(enc4)

        # Decoder with skip connections
        x = self.up_conv5(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.conv5(x)

        x = self.up_conv6(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.conv8(x)

        x = self.final_conv(x)

        return x
