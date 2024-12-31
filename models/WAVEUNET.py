import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=9, stride=1, padding="same"):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=channel_in, 
                      out_channels=channel_out, 
                      kernel_size=kernel_size,
                      stride=stride, 
                      padding=padding, 
                      dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.main(x)
        return self.dropout(x)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=9, stride=1, padding="same"):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=channel_in, 
                      out_channels=channel_out, 
                      kernel_size=kernel_size,
                      stride=stride, 
                      padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)
    

class WaveUNetPart(nn.Module):
    def __init__(self, n_layers=8, channels_interval=16):
        super(WaveUNetPart, self).__init__()
        self.n_layers = n_layers
        self.channels_interval = channels_interval

        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, kernel_size=3, stride=1,
                      padding="same"),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]

        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(in_channels=1+self.channels_interval, out_channels=1, kernel_size=1, stride=1),
            nn.Tanh()
        )


    def forward(self, x):
        tmp = []
        o = x
        # print("1", o.size)
        # Up Sample
        for encoder in self.encoder:
            # print("******")
            o = encoder(o)
            # print("**",o.shape)
            tmp.append(o)
            o = F.max_pool1d(o, kernel_size=2, stride=2)
            # print("**",o.shape)

        o = self.bottleneck(o)
        # print("bottleneck",o.shape)

        for i in range(self.n_layers):
            # print("******")
            o = F.interpolate(o, size = tmp[self.n_layers - i - 1].shape[-1], mode="linear", align_corners=True)
            # print("**",o.shape)
            # print("**", tmp[self.n_layers - i - 1].shape)
            o = torch.cat((o, tmp[self.n_layers - i - 1]), dim=1)
            # print("**",o.shape)
            o = self.decoder[i](o)
            # print("**",o.shape)
        o = torch.cat((o, x), dim=1)
        # print(o.shape)
        # print(1+self.channels_interval)
        o = self.out(o)
        # print(o.shape)
        return o
    

class WaveUNet(nn.Module):

    def __init__(self, n_layers=8, channels_interval=16):
        super(WaveUNet, self).__init__()
        
        self.firstUNet = WaveUNetPart(n_layers=n_layers, channels_interval=channels_interval)
        self.secondUNet = WaveUNetPart(n_layers=n_layers, channels_interval=channels_interval)

    def forward(self, x):
        # x = [B, 1, 80000]
        audio_firstpart = x[:,:,:50000]
        audio_secondpart = x[:,:,-50000:]
        
        audio_firstpart = self.firstUNet(audio_firstpart)[:,:,:40000]
        audio_secondpart = self.secondUNet(audio_secondpart)[:,:,-40000:]

        return torch.cat([audio_firstpart, audio_secondpart], dim=2)