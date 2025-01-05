import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.VisionTransformer import VisionTransformerClassifier
from torchvision.transforms import v2
from torchvision import transforms
import random

class ConvAttentionProj(nn.Module):
    def __init__(self, channels: int, groups: int, channel_scale: int = 1):
        super().__init__()
        self.channels = channels
        self.pointwise = nn.Conv2d(self.channels,
                                   self.channels * groups * channel_scale,
                                   kernel_size=1)
        self.depthwise = nn.Conv2d(self.channels * groups * channel_scale,
                                   self.channels * groups * channel_scale,
                                   kernel_size=3,
                                   padding=1,
                                   groups=self.channels)
        self.groups = groups

    def forward(self, x: torch.Tensor):
        x = self.depthwise(self.pointwise(x))
        return x.chunk(self.groups, dim=1)


class MDTAttention(nn.Module):
    def __init__(self, height: int, width: int, channels: int, num_heads: int):
        super().__init__()
        assert channels % num_heads == 0
        self.height = height
        self.width = width
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.pointwise_w = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.norm = nn.LayerNorm([channels, height, width])
        self.scaling_factor = nn.Parameter(torch.tensor(math.sqrt(self.head_channels)))
        self.dconv_proj = ConvAttentionProj(channels, 3)


    def forward(self, x: torch.Tensor):
        # input x is of shape (B=batch, C=channels, H=height, W=width)
        B, C, H, W = x.shape
        Nh = self.num_heads
        Hc = self.head_channels
        x = self.norm(x)
        Q, K, V = self.dconv_proj(x)  # each is (B, C, H, W)
        Q = Q.reshape(B, Nh, Hc, -1)  # (B, Nh, Hc, H*W)
        K = K.reshape(B, Nh, Hc, -1)  # (B, Nh, Hc, H*W)
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)
        V = V.reshape(B, Nh, Hc, -1)  # (B, Nh, Hc, H*W)
        A_unnormalized = (Q @ K.permute(0, 1, 3, 2) / self.scaling_factor)  # (B, Nh, Hc, Hc)
        A_T = F.softmax(A_unnormalized, dim=-1).permute(0, 1, 3, 2)  # (B, Nh, Hc, Hc)
        VA = A_T @ V  # (B, Nh, Hc, H*W)
        VA = VA.reshape(B, C, H*W).reshape(B, C, H, W)  # (B, C, H, W)
        return x + self.pointwise_w(VA)


class GDFNet(nn.Module):
    def __init__(self, height: int, width: int, channels: int, channel_scale: int = 4, out_channels=None):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.norm = nn.LayerNorm([channels, height, width])
        self.channel_scale = channel_scale
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = channels
        hidden_channel_dim = channel_scale * channels
        self.dconv_proj = ConvAttentionProj(channels, 2, channel_scale)
        self.pointwise_w = nn.Conv2d(hidden_channel_dim, channels, kernel_size=1)
        self.pointwise_out = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        out_1, out_2 = self.dconv_proj(x)
        gated = self.gelu(out_1) * out_2
        return self.pointwise_out(self.pointwise_w(gated) + x)


class RestormerBlock(nn.Module):
    def __init__(self, num_heads: int, height: int, width: int, channels: int, channel_scale: int = 4, out_channels = None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else channels
        self.mdtattention = MDTAttention(height, width, channels, num_heads)
        self.gdfnet = GDFNet(height, width, channels, channel_scale, out_channels)

    def forward(self, x: torch.Tensor):
        x = self.mdtattention(x)
        x = self.gdfnet(x)
        return x


class WPM(nn.Module):
    def __init__(self, num_layers: int, num_bits: int, out_channels: int, channel_scale: int = 4):
        super().__init__()
        size = int(math.sqrt(num_bits))
        self.num_bits = num_bits
        self.size = size
        assert size * size == num_bits
        blocks = []
        for i in range(num_layers - 1):
            scale = 2 ** (i+1)
            blocks.append(nn.PixelShuffle(2))
            blocks.append(RestormerBlock(1, size*scale, size*scale, 1, channel_scale, 4))
        scale = 2 ** num_layers
        blocks.append(nn.PixelShuffle(2))
        blocks.append(RestormerBlock(1, size*scale, size*scale, 1, channel_scale, out_channels))
        self.transformer_blocks = nn.Sequential(*blocks)

    def forward(self, watermarks: torch.Tensor):
        B, _ = watermarks.shape
        wm = watermarks.reshape(B, 1, self.size, self.size).repeat(1, 4, 1, 1)
        wm = self.transformer_blocks(wm)
        return wm


class FEM(nn.Module):
    def __init__(self, height: int, width: int, channels: int, num_heads: int):
        super().__init__()
        assert channels % num_heads == 0
        self.height = height
        self.width = width
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.image_dconv_proj = ConvAttentionProj(channels, 2)
        self.wm_dconv_proj = ConvAttentionProj(channels, 1)
        self.scaling_factor = nn.Parameter(torch.tensor(math.sqrt(self.head_channels)))
        self.pointwise_w = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.image_norm = nn.LayerNorm([channels, height, width])
        self.wm_norm = nn.LayerNorm([channels, height, width])

    def forward(self, image, watermark):
        image = self.image_norm(image)
        watermark = self.wm_norm(watermark)
        B, C, H, W = image.shape
        Nh = self.num_heads
        Hc = self.head_channels
        K, V = self.image_dconv_proj(image)
        Q = self.wm_dconv_proj(watermark)[0]
        Q = Q.reshape(B, Nh, Hc, H*W)  # (B, Nh, Hc, H*W)
        K = K.reshape(B, Nh, Hc, H*W)  # (B, Nh, Hc, H*W)
        V = V.reshape(B, Nh, Hc, H*W)  # (B, Nh, Hc, H*W)
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)
        A_unnormalized = (Q @ K.permute(0, 1, 3, 2) / self.scaling_factor)  # (B, Nh, Hc, Hc)
        A_T = F.softmax(A_unnormalized, dim=-1).permute(0, 1, 3, 2)  # (B, Nh, Hc, Hc)
        VA = A_T @ V  # (B, Nh, Hc, H*W)
        VA = VA.reshape(B, C, H*W).reshape(B, C, H, W)  # (B, C, H, W)
        return image + self.pointwise_w(VA)


class SFM(nn.Module):
    def __init__(self, height: int, width: int, channels: int, num_heads: int):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.image_dconv_proj = ConvAttentionProj(channels, 3)
        self.wm_dconv_proj = ConvAttentionProj(channels, 3)
        self.image_norm = nn.LayerNorm([channels, height, width])
        self.wm_norm = nn.LayerNorm([channels, height, width])
        self.proj = nn.Parameter(torch.empty(size=(2*channels, channels)))
        nn.init.kaiming_uniform_(self.proj, a=math.sqrt(5))

    def forward(self, image: torch.Tensor, watermark: torch.Tensor):
        B, C, H, W = image.shape
        image = self.image_norm(image)
        watermark = self.wm_norm(watermark)
        Nh = self.num_heads
        Hc = self.head_channels
        Qimage, Kimage, Vimage = self.image_dconv_proj(image)
        Qwm, Kwm, Vwm = self.wm_dconv_proj(watermark)
        Qimage = Qimage.reshape(B, Nh, Hc, -1)  # (B, Nh, Hc, H*W)
        Kimage = Kimage.reshape(B, Nh, Hc, -1)  # (B, Nh, Hc, H*W)
        F.normalize(Qimage, dim=-1)
        F.normalize(Kimage, dim=-1)
        Vimage = Vimage.reshape(B, Nh, Hc, -1)  # (B, Nh, Hc, H*W)
        Qwm = Qwm.reshape(B, Nh, Hc, -1)  # (B, Nh, Hc, H*W)
        Kwm = Kwm.reshape(B, Nh, Hc, -1)  # (B, Nh, Hc, H*W)
        F.normalize(Qwm, dim=-1)
        F.normalize(Kwm, dim=-1)
        Vwm = Vwm.reshape(B, Nh, Hc, -1)  # (B, Nh, Hc, H*W)
        V_concat = torch.cat((Vimage, Vwm), dim=2)  # (B, Nh, 2*Hc, H*W)
        K_concat = torch.cat((Kimage, Kwm), dim=2)  # (B, Nh, 2*Hc, H*W)
        A_image = F.softmax(K_concat @ Qimage.permute(0, 1, 3, 2), dim=-1)  # (B, Nh, 2*Hc, Hc)
        A_wm = F.softmax(K_concat @ Qwm.permute(0, 1, 3, 2), dim=-1)  # (B, Nh, 2*Hc, Hc)
        image_attn = A_image.permute(0, 1, 3, 2) @ V_concat  # (B, Nh, Hc, H*W)
        wm_attn = A_wm.permute(0, 1, 3, 2) @ V_concat   # (B, Nh, Hc, H*W)
        attn_concat = torch.cat((image_attn, wm_attn), dim=2)   # (B, Nh, 2*Hc, H*W)
        attn_concat = attn_concat.reshape(B, 2*C, H*W).reshape(B, 2*C, H, W)  # (B, 2*C, H, W)
        return (attn_concat.permute(0, 2, 3, 1) @ self.proj).permute(0, 3, 1, 2)


class Encoder(nn.Module):
    def __init__(self, image_size: int, bit_length: int, num_fems: int, hidden_channels: int, num_heads: int):
        super().__init__()
        self.image_size = image_size
        self.bit_length = bit_length
        assert int(math.sqrt(bit_length)) ** 2 == bit_length
        assert math.log2(image_size) == int(math.log2(image_size))
        assert math.log2(bit_length) == int(math.log2(bit_length))
        assert hidden_channels % num_heads == 0
        self.num_blocks = int(math.log2(image_size / math.sqrt(bit_length)))
        self.in_conv = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.fems = nn.ModuleList([FEM(image_size, image_size, hidden_channels, num_heads) for _ in range(num_fems)])
        self.sfm = SFM(image_size, image_size, hidden_channels, num_heads)
        self.wpm = WPM(self.num_blocks, bit_length, hidden_channels)
        self.out_transformer = RestormerBlock(num_heads, image_size, image_size, hidden_channels)
        self.out_conv = nn.Conv2d(hidden_channels, out_channels=3, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor, watermarks: torch.Tensor):
        Fc = self.in_conv(images)
        Fw = self.wpm(watermarks)
        Ff = Fc
        for f in self.fems:
            Ff = f(Ff, Fw)
        Fs = self.sfm(Ff, Fw)
        Fs_attn = self.out_transformer(Fs)
        return self.out_conv(Fs_attn) + images


class Decoder(nn.Module):
    def __init__(self, image_size: int, bit_length: int, hidden_channels: int):
        super().__init__()
        self.image_size = image_size
        self.bit_length = bit_length
        assert int(math.sqrt(bit_length)) ** 2 == bit_length
        assert math.log2(image_size) == int(math.log2(image_size))
        assert math.log2(bit_length) == int(math.log2(bit_length))
        assert hidden_channels % 2 == 0
        self.num_blocks = int(math.log2(image_size / math.sqrt(bit_length)))
        self.in_conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=3, padding=1)
        blocks = []

        for i in range(self.num_blocks-1):
            downscale = 2 ** (i+1)
            blocks.append(nn.PixelUnshuffle(2))
            blocks.append(RestormerBlock(1, int(image_size/downscale), int(image_size/downscale), 16, out_channels=4))
        downscale = 2 ** self.num_blocks
        blocks.append(nn.PixelUnshuffle(2))
        blocks.append(RestormerBlock(1, int(image_size/downscale), int(image_size/downscale), 16, out_channels=hidden_channels))
        self.transformer_blocks = nn.Sequential(*blocks)

    def forward(self, images: torch.Tensor):
        B, C, H, W = images.shape
        x = self.in_conv(images)
        x = self.transformer_blocks(x)
        x = self.out_conv(x)
        return F.sigmoid(x.reshape(B, self.bit_length))