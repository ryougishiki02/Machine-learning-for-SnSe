import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model

class crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-1, 0:W]
        return x

class shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_down = nn.ZeroPad2d((0,0,1,0))
            self.crop = crop()
        self.replicate = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=bias)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.blind:
            x = self.shift_down(x)
        x = self.replicate(x)
        x = self.conv(x)
        x = self.relu(x)
        if self.blind:
            x = self.crop(x)
        return x

class Pool(nn.Module):
    def __init__(self, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        x = self.pool(x)
        return x

class rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2,3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2,3).flip(2)
        x = torch.cat((x,x90,x180,x270), dim=0)
        return x

class unrotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2,3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2,3).flip(3)
        x = torch.cat((x0,x90,x180,x270), dim=1)
        return x

class ENC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, reduce=True, blind=True):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, out_channels, bias=bias, blind=blind)
        if reduce:
            self.pool = Pool(blind=blind)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.reduce:
            x = self.pool(x)
        return x

class DEC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, blind=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv4 = Conv(mid_channels, out_channels, bias=bias, blind=blind)

    def forward(self, x, x_in):
        x = self.upsample(x)

        # Smart Padding
        diffY = x_in.size()[2] - x.size()[2]
        diffX = x_in.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Blind_UNet(nn.Module):
    def __init__(self, n_channels=3, n_output=96, bias=False, blind=True):
        super().__init__()
        self.n_channels = n_channels
        self.bias = bias
        self.enc1 = ENC_Conv(n_channels, 48, 48, bias=bias, blind=blind)
        self.enc2 = ENC_Conv(48, 48, 48, bias=bias, blind=blind)
        self.enc3 = ENC_Conv(48, 96, 48, bias=bias, reduce=False, blind=blind)
        self.dec2 = DEC_Conv(96, 96, 96, bias=bias, blind=blind)
        self.dec1 = DEC_Conv(96+n_channels, 96, n_output, bias=bias, blind=blind)

    def forward(self, input):
        x1 = self.enc1(input)
        x2 = self.enc2(x1)
        x = self.enc3(x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        return x

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv1x1 = nn.Conv2d(in_channels*3, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        out = torch.cat([conv1_out, conv3_out, conv5_out], dim=1)
        out = self.conv1x1(out)
        return out


class NonLocalAttention(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        g = self.g(x).view(batch_size, self.inter_channels, -1)
        g = g.permute(0, 2, 1)
        theta = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta = theta.permute(0, 2, 1)
        phi = self.phi(x).view(batch_size, self.inter_channels, -1)
        attention = torch.matmul(theta, phi)
        attention = F.softmax(attention, dim=-1)

        print("Shape of attention:", attention.shape)
        print("Shape of g:", g.shape)

        # 确保 attention 和 g 的维度匹配
        g_permuted = g.permute(0, 2, 1)
        if attention.shape[-1] != g_permuted.shape[-2]:
            raise ValueError(f"Dimensions do not match for matmul: {attention.shape} and {g_permuted.shape}")

        out = torch.matmul(attention, g_permuted).permute(0, 2, 1).view(batch_size, self.inter_channels, H, W)
        out = self.W(out)
        return out + x

class MANBA(nn.Module):
    def __init__(self, in_channels):
        super(MANBA, self).__init__()
        self.msa = MultiScaleAttention(in_channels)
        self.nla = NonLocalAttention(in_channels)

    def forward(self, x):
        x = self.msa(x)
        x = self.nla(x)
        return x

@register_model("blind-spot-net-4")
class BlindSpotNet(nn.Module):
    def __init__(self, n_channels=3, n_output=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.n_channels = n_channels
        self.c = n_channels
        self.n_output = n_output
        self.bias = bias
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.unet = Blind_UNet(n_channels=n_channels, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = Blind_UNet(n_channels=n_channels, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, n_output, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--in-channels", type=int, default=3, help="number of input channels")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(n_channels=args.in_channels, n_output=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        x = self.rotate(x)
        x = self.unet(x)
        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x, sigma

@register_model("blind-video-net-d1-4")
class BlindVideoNetD1(nn.Module):
    def __init__(self, channels_per_frame=3, out_channels=9, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = Blind_UNet(n_channels=3*channels_per_frame, n_output=96, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = Blind_UNet(n_channels=3*channels_per_frame, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=3, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=9, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        x = self.rotate(x)
        x = self.denoiser_1(x)
        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x, sigma

@register_model("blind-video-net-4")
class BlindVideoNet(nn.Module):
    def __init__(self, channels_per_frame=1, out_channels=1, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = Blind_UNet(n_channels=3*channels_per_frame, n_output=32, bias=bias, blind=blind)
        self.denoiser_2 = Blind_UNet(n_channels=96, n_output=96, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = Blind_UNet(n_channels=5*channels_per_frame, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=1, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=1, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        i1 = self.rotate(x[:, 0:(3*self.c), :, :])
        i2 = self.rotate(x[:, self.c:(4*self.c), :, :])
        i3 = self.rotate(x[:, (2*self.c):(5*self.c), :, :])

        y1 = self.denoiser_1(i1)
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3)

        y = torch.cat((y1, y2, y3), dim=1)
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x, sigma

@register_model("manba-blind-video")
class ManbaBlindVideo(nn.Module):
    def __init__(self, channels_per_frame=1, out_channels=1, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = Blind_UNet(n_channels=3*channels_per_frame, n_output=32, bias=bias, blind=blind)
        self.manba = MANBA(32)  # Add MANBA module
        self.denoiser_2 = Blind_UNet(n_channels=96, n_output=96, bias=bias, blind=blind)
        if not sigma_known:
            self.sigma_net = Blind_UNet(n_channels=5*channels_per_frame, n_output=1, bias=False, blind=False)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--channels", type=int, default=1, help="number of channels per frame")
        parser.add_argument("--out-channels", type=int, default=1, help="number of output channels")
        parser.add_argument("--bias", action='store_true', help="use residual bias")
        parser.add_argument("--normal", action='store_true', help="not a blind network")
        parser.add_argument("--blind-noise", action='store_true', help="noise sigma is not known")

    @classmethod
    def build_model(cls, args):
        return cls(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise))

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        if not self.sigma_known:
            sigma = self.sigma_net(x).mean(dim=(1,2,3))
        else:
            sigma = None

        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        i1 = self.rotate(x[:, 0:(3*self.c), :, :])
        i2 = self.rotate(x[:, self.c:(4*self.c), :, :])
        i3 = self.rotate(x[:, (2*self.c):(5*self.c), :, :])

        y1 = self.denoiser_1(i1)
        y1 = self.manba(y1)  # Apply MANBA after first denoising step
        y2 = self.denoiser_1(i2)
        y2 = self.manba(y2)  # Apply MANBA after first denoising step
        y3 = self.denoiser_1(i3)
        y3 = self.manba(y3)  # Apply MANBA after first denoising step

        y = torch.cat((y1, y2, y3), dim=1)  # Concatenate features from MANBA
        x = self.denoiser_2(y)

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x)
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)

        # Unsquare
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]
        return x, sigma