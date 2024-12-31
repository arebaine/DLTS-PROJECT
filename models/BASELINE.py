from models.utils import EncoderConvBlock, DecoderConvBlock
import torch as torch
import torch.nn.functional as F


class BASELINE(torch.nn.Module):
    def __init__(
        self, n_input_channel=1, n_output_channel=1, kernel_size=5, stride=2, padding=0
    ):
        super().__init__()
        self.n_input_channel = n_input_channel
        self.n_output_channel = n_output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = EncoderConvBlock(self.n_input_channel, 16, 5, 2, 0)
        self.conv2 = EncoderConvBlock(16, 32, 5, 2, 0)
        self.conv3 = EncoderConvBlock(32, 64, 5, 2, 0)
        self.conv4 = EncoderConvBlock(64, 128, 5, 2, 0)
        self.conv5 = EncoderConvBlock(128, 256, 5, 2, 0)
        self.conv6 = EncoderConvBlock(256, 512, 5, 2, 0)

        self.dec1 = DecoderConvBlock(512, 256, 5, 2, 0, dropout=True)
        self.dec2 = DecoderConvBlock(256, 128, 5, 2, 0, dropout=True)
        self.dec3 = DecoderConvBlock(128, 64, 5, 2, 0, dropout=True)
        self.dec4 = DecoderConvBlock(64, 32, 5, 2, 0, dropout=False)
        self.dec5 = DecoderConvBlock(32, 16, 5, 2, 0, dropout=False)
        self.dec6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                16, self.n_output_channel, kernel_size=5, stride=2
            ),
            torch.nn.BatchNorm2d(self.n_output_channel),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        _, out1 = self.conv1(x)
        _, out2 = self.conv2(out1)
        _, out3 = self.conv3(out2)
        _, out4 = self.conv4(out3)
        _, out5 = self.conv5(out4)
        _, out6 = self.conv6(out5)

        dec1 = self.dec1(out6)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(F.pad(dec2, (1, 2, 1, 2), "constant", 0)[:, :, 1:-1, 1:-1])
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(F.pad(dec4, (1, 2, 1, 2), "constant", 0)[:, :, 1:-1, 1:-1])
        dec6 = self.dec6(dec5)

        return x * dec6