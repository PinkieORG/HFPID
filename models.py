import torch.nn as nn


class EIU(nn.Module):
    def __init__(self, in_channels=3, dims=None):
        super().__init__()
        if dims is None:
            dims = [32, 64, 64, 64, 32]
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(dims[0], dims[1], 3, padding=1), nn.LeakyReLU(),
            nn.ConvTranspose2d(dims[1], dims[2], 4, stride=2, padding=1),
            nn.Conv2d(dims[2], dims[3], 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(dims[3], dims[4], 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(dims[4], in_channels, 3, padding=1))
        self.bilinear = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)

    def forward(self, x):
        return self.layers(x) + self.bilinear(x)


class DID(nn.Module):
    def __init__(self, in_channels=3, dims=None):
        super().__init__()
        if dims is None:
            dims = [32, 32]
        self.space_to_depth = nn.PixelUnshuffle(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(4 * in_channels, dims[0], 3, padding=1), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(dims[0], dims[0], 3, padding=1),
                                   nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(dims[0], dims[1], 3, padding=1),
                                   nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(dims[1], dims[1], 3, padding=1),
                                   nn.LeakyReLU())
        self.conv5 = nn.Conv2d(dims[1], in_channels, 3, padding=1)

    def forward(self, x):
        y = self.space_to_depth(x)
        y = self.conv1(y)
        y = self.conv2(y) + y
        y = self.conv3(y)
        y = self.conv4(y) + y
        y = self.conv5(y)
        return y + nn.functional.interpolate(x, scale_factor=0.5)
