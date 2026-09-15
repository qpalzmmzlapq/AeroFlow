import torch
import torch.nn as nn
import torch.nn.functional as F


class AeroNet(nn.Module):
    class TwoConv(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.Convlayer = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.Convlayer(x)

    class Encoder(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.Conv = AeroNet.TwoConv(in_c, out_c)
            self.maxpool = nn.MaxPool2d(2)
        def forward(self, x):
            x = self.Conv(x)
            y = self.maxpool(x)
            return y, x

    class Decoder(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
            self.conv = AeroNet.TwoConv(in_c, out_c)
        def forward(self, x, y):
            x = self.transpose(x)
            u = torch.cat([x, y], dim=1)
            u = self.conv(u)
            return u

    def __init__(self, in_c=3, num_c=3, base=32):
        super().__init__()

        # Общий encoder
        self.enc1 = self.Encoder(in_c, base)
        self.enc2 = self.Encoder(base, base * 2)
        self.enc3 = self.Encoder(base * 2, base * 4)
        self.enc4 = self.Encoder(base * 4, base * 8)

        # Три bottleneck
        self.bottleneck_ux = self.TwoConv(base * 8, base * 16)
        self.bottleneck_uy = self.TwoConv(base * 8, base * 16)
        self.bottleneck_p  = self.TwoConv(base * 8, base * 16)

        # Три декодера
        def make_decoder():
            return nn.ModuleList([
                self.Decoder(base * 16, base * 8),
                self.Decoder(base * 8, base * 4),
                self.Decoder(base * 4, base * 2),
                self.Decoder(base * 2, base),
            ])
        self.dec_ux = make_decoder()
        self.dec_uy = make_decoder()
        self.dec_p  = make_decoder()

        # Три выходные головы
        self.out_ux = nn.Conv2d(base, 1, 1)
        self.out_uy = nn.Conv2d(base, 1, 1)
        self.out_p  = nn.Conv2d(base, 1, 1)

    def _run_decoder(self, x, decoder, skips):
        for dec, skip in zip(decoder, skips[::-1]):
            x = dec(x, skip)
        return x

    def forward(self, x):
        x_in = x

        # Общий encoder
        x, y1 = self.enc1(x)
        x, y2 = self.enc2(x)
        x, y3 = self.enc3(x)
        x, y4 = self.enc4(x)
        skips = [y1, y2, y3, y4]

        # Три ветви
        d_ux = self._run_decoder(self.bottleneck_ux(x), self.dec_ux, skips)
        d_uy = self._run_decoder(self.bottleneck_uy(x), self.dec_uy, skips)
        d_p  = self._run_decoder(self.bottleneck_p(x),  self.dec_p,  skips)

        ux = self.out_ux(d_ux)
        uy = self.out_uy(d_uy)
        p  = self.out_p(d_p)

        out = torch.cat([ux, uy, p], dim=1)

        # Hard BC
        mask_flow = (x_in[:, 1:2] != 0).float()
        ux_f = out[:, 0:1] * mask_flow
        uy_f = out[:, 1:2] * mask_flow
        p_f  = out[:, 2:3]
        return torch.cat([ux_f, uy_f, p_f], dim=1)