from models.utils import EncoderConvBlock, DecoderConvBlock
import torch as torch
import torch.nn.functional as F


class UNet(torch.nn.Module):
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
        self.dec2 = DecoderConvBlock(512, 128, 5, 2, 0, dropout=True)
        self.dec3 = DecoderConvBlock(256, 64, 5, 2, 0, dropout=True)
        self.dec4 = DecoderConvBlock(128, 32, 5, 2, 0, dropout=False)
        self.dec5 = DecoderConvBlock(64, 16, 5, 2, 0, dropout=False)
        self.dec6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                32, self.n_output_channel, kernel_size=5, stride=2
            ),
            torch.nn.BatchNorm2d(self.n_output_channel),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        raw1, out1 = self.conv1(x)
        raw2, out2 = self.conv2(out1)
        raw3, out3 = self.conv3(out2)
        raw4, out4 = self.conv4(out3)
        raw5, out5 = self.conv5(out4)
        raw6, out6 = self.conv6(out5)

        dec1 = self.dec1(out6)
        dec2 = self.dec2(torch.cat((dec1, raw5), dim=1))
        dec3 = self.dec3(
            torch.cat(
                (F.pad(dec2, (1, 2, 1, 2), "constant", 0)[:, :, 1:-1, 1:-1], raw4),
                dim=1,
            )
        )
        dec4 = self.dec4(torch.cat((dec3, raw3), dim=1))
        dec5 = self.dec5(
            torch.cat(
                (F.pad(dec4, (1, 2, 1, 2), "constant", 0)[:, :, 1:-1, 1:-1], raw2),
                dim=1,
            )
        )
        dec6 = self.dec6(torch.cat((dec5, raw1), dim=1))

        return x * dec6