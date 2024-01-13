import torch
import torch.nn as nn
from collections import OrderedDict


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80
        n = self.num_features

        ########################################################################################
        # 4 Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=n, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv3d(in_channels=n, out_channels=n*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(n * 2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.enc3 = nn.Sequential(
            nn.Conv3d(in_channels=n*2, out_channels=n*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(n * 4),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.enc4 = nn.Sequential(
            nn.Conv3d(in_channels=n*4, out_channels=n*8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(n*8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 2 Bottleneck layers
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=n*8, out_features=n*8),
            nn.ReLU(),
            nn.Linear(in_features=n*8, out_features=n*8),
            nn.ReLU(),
        )

        # 4 Decoder layers
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=n*8*2, out_channels=n*4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(n*4),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=n*4*2, out_channels=n*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(n*2),
            nn.ReLU(),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=n*2*2, out_channels=n, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(n),
            nn.ReLU(),
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=n*2, out_channels=1, kernel_size=4, stride=2, padding=1),
        )
        ########################################################################################

    def forward(self, x):
        b = x.shape[0]

        ########################################################################################
        # Encode
        # Pass x though encoder while keeping the intermediate outputs for the skip connections
        x_e1 = self.enc1(x)    # 2*32*32*32  -> 80*16*16*16
        x_e2 = self.enc2(x_e1) # 80*16*16*16 -> 160*8*8*8
        x_e3 = self.enc3(x_e2) # 160*8*8*8   -> 320*4*4*4
        x_e4 = self.enc4(x_e3) # 320*4*4*4   -> 640*1*1*1

        # Bottleneck
        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)

        # Decode
        # Pass x through the decoder, applying the skip connections in the process
        x_d1 = self.dec1(torch.cat((x, x_e4), dim=1))     # (640*1*1*1)*2   -> 320*4*4*4
        x_d2 = self.dec2(torch.cat((x_d1, x_e3), dim=1))  # (320*4*4*4)*2   -> 160*8*8*8
        x_d3 = self.dec3(torch.cat((x_d2, x_e2), dim=1))  # (160*8*8*8)*2   -> 80*16*16*16
        x_d4 = self.dec4(torch.cat((x_d3, x_e1), dim=1))  # (80*16*16*16)*2 -> 1*32*32*32
        ########################################################################################

        x = torch.squeeze(x_d4, dim=1)

        ########################################################################################
        # Log scaling
        x = torch.log(torch.abs(x) + 1)
        ########################################################################################

        return x
