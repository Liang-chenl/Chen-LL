# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import time
#
# from PIL.ImageOps import expand
# from scipy.io import savemat
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
# from einops import rearrange, repeat
# from functools import partial
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from pdb import set_trace as stx
# from typing import Optional, Callable
# import math
# import numbers
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# import sys
#
# from torch.fft import ifft2
# from torch.nn.functional import relu_
#
# from basicsr.utils.registry import ARCH_REGISTRY
# import torch.autograd
# import numpy as np
# import pywt  # 假设使用pywt库进行曲波变换
#
# class Attention(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
#         super(Attention, self).__init__()
#         attention_channel = max(int(in_planes * reduction), min_channel)
#         self.kernel_size = kernel_size
#         self.kernel_num = kernel_num
#         self.temperature = 1.0
#
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
#         self.relu = nn.GELU()
#
#         self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
#         self.func_channel = self.get_channel_attention
#
#         if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
#             self.func_filter = self.skip
#         else:
#             self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
#             self.func_filter = self.get_filter_attention
#
#         if kernel_size == 1:  # point-wise convolution
#             self.func_spatial = self.skip
#         else:
#             self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
#             self.func_spatial = self.get_spatial_attention
#
#         if kernel_num == 1:
#             self.func_kernel = self.skip
#         else:
#             self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
#             self.func_kernel = self.get_kernel_attention
#
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def update_temperature(self, temperature):
#         self.temperature = temperature
#
#     @staticmethod
#     def skip(_):
#         return 1.0
#
#     def get_channel_attention(self, x):
#         channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
#         return channel_attention
#
#     def get_filter_attention(self, x):
#         filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
#         return filter_attention
#
#     def get_spatial_attention(self, x):
#         spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
#         spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
#         return spatial_attention
#
#     def get_kernel_attention(self, x):
#         kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
#         kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
#         return kernel_attention
#
#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.fc(x)
#         x = self.relu(x)
#         return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
#
#
# class Lap_Pyramid_Conv(nn.Module):
#     def __init__(self, dim):
#         super(Lap_Pyramid_Conv, self).__init__()
#         self.dim = dim
#         self.kernel = None  # 初始化为空，动态生成卷积核
#
#     def gauss_kernel(self, channels, device=torch.device('cuda')):
#         """
#         根据通道数动态生成高斯卷积核
#         """
#         kernel = torch.tensor([[1., 4., 6., 4., 1],
#                                [4., 16., 24., 16., 4.],
#                                [6., 24., 36., 24., 6.],
#                                [4., 16., 24., 16., 4.],
#                                [1., 4., 6., 4., 1.]])
#         kernel /= 256.
#         kernel = kernel.repeat(channels, 1, 1, 1)  # 根据通道数扩展
#         kernel = kernel.to(device)
#         return kernel
#
#     def downsample(self, x):
#         return x[:, :, ::2, ::2]
#
#     def upsample(self, x):
#         """
#         上采样并进行高斯平滑
#         """
#         cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
#         cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
#         cc = cc.permute(0, 1, 3, 2)
#         cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
#         cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
#         x_up = cc.permute(0, 1, 3, 2)
#
#         # 动态生成卷积核
#         kernel = self.gauss_kernel(x_up.shape[1], x_up.device)
#         return self.conv_gauss(x_up, 4 * kernel)
#
#     def conv_gauss(self, img, kernel):
#         """
#         高斯卷积，动态调整通道数
#         """
#         img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
#
#         # 检查卷积核与输入通道数是否匹配
#         if kernel.shape[0] != img.shape[1]:
#             kernel = kernel[:img.shape[1], :, :, :]  # 动态调整核大小
#
#         out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
#         return out
#
#     def decompose(self, img):
#         """
#         分解图像为一个高频分量和一个低频分量
#         """
#         # 动态生成卷积核
#         if self.kernel is None or self.kernel.shape[0] != img.shape[1]:
#             self.kernel = self.gauss_kernel(img.shape[1], img.device)
#
#         filtered = self.conv_gauss(img, self.kernel)
#         low_freq = self.downsample(filtered)
#         upsampled_low = self.upsample(low_freq)
#
#         # 调整尺寸匹配
#         if upsampled_low.shape[2] != img.shape[2] or upsampled_low.shape[3] != img.shape[3]:
#             upsampled_low = nn.functional.interpolate(upsampled_low, size=(img.shape[2], img.shape[3]))
#
#         high_freq = img - upsampled_low
#         return high_freq, low_freq
#
#     def reconstruct(self, high_freq, low_freq):
#         """
#         使用高频分量和低频分量重建图像
#         """
#         upsampled_low = self.upsample(low_freq)
#
#         # 调整尺寸匹配
#         if upsampled_low.shape[2] != high_freq.shape[2] or upsampled_low.shape[3] != high_freq.shape[3]:
#             upsampled_low = nn.functional.interpolate(upsampled_low, size=(high_freq.shape[2], high_freq.shape[3]))
#
#         reconstructed = upsampled_low + high_freq
#         return reconstructed
#
#
#
# class FFT(nn.Module):
#     def __init__(self,dim):
#         super(FFT, self).__init__()
#
#         # 初始化拉普拉斯金字塔模块
#         self.lap_pyramid = Lap_Pyramid_Conv(dim)
#
#         # 自动将模型迁移至 GPU（如果可用）
#         if torch.cuda.is_available():
#             self.cuda()  # 将模型迁移到 GPU
#
#     def forward(self, x):
#         # 调用拉普拉斯分频函数
#         high_freq, low_freq = self.lap_pyramid.decompose(x)
#
#         # 返回分解后的高频和低频分量
#         return high_freq, low_freq
#
#
# class IFFT(nn.Module):
#     def __init__(self,dim):
#         super(IFFT, self).__init__()
#         self.lap_pyramid = Lap_Pyramid_Conv(dim)
#
#     def forward(self, low_freq, high_freq):
#         reconstructed_image = self.lap_pyramid.reconstruct(high_freq, low_freq)
#         return reconstructed_image
#
#
#
# class ChannelAttention(nn.Module):
#     """Channel attention used in RCAN.
#     Args:
#         num_feat (int): Channel number of intermediate features.
#         squeeze_factor (int): Channel squeeze factor. Default: 16.
#     """
#
#     def __init__(self, num_feat, squeeze_factor=16):
#         super(ChannelAttention, self).__init__()
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
#             nn.Sigmoid())
#
#     def forward(self, x):
#         y = self.attention(x)
#         return x * y
#
#
# class CAB(nn.Module):
#
#     def __init__(self, num_feat, compress_ratio=3, squeeze_factor=16):
#         super(CAB, self).__init__()
#
#         self.cab = nn.Sequential(
#             nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
#             nn.GELU(),
#             nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
#             ChannelAttention(num_feat, squeeze_factor)
#             )
#
#     def forward(self, x):
#         return self.cab(x)
#
#
# class SimpleGate(nn.Module):
#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=1)
#         return F.gelu(x1)*x2
#
# class ffn(nn.Module):
#     def __init__(self,
#                  num_feat,
#                  norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),):
#         super(ffn, self).__init__()
#         self.ln_1 = norm_layer(num_feat)
#         self.dw_conv = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, groups=num_feat)
#         self.pw_conv = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0)
#         self.sg = SimpleGate()
#         self.conv3 = nn.Conv2d(num_feat//2,num_feat,1,1,0)
#         self.skip_scale2 = nn.Parameter(torch.ones(1, num_feat, 1, 1))
#
#
#     def forward(self, input):
#         x = self.ln_1(input.permute(0, 2, 3, 1).contiguous())
#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = self.dw_conv(x)  # DW 卷积
#         x = self.pw_conv(x)
#         x = self.sg(x)
#         x = self.conv3(x)+input*self.skip_scale2
#         return x
#
# class SS2D(nn.Module):
#     def __init__(
#             self,
#             d_model,
#             d_state=16,
#             d_conv=3,
#             expand=2,
#             dt_rank="auto",
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             dropout=0.,
#             conv_bias=True,
#             bias=False,
#             device=None,
#             dtype=None,
#             **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#
#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
#         self.conv2d = nn.Conv2d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             groups=self.d_inner,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#             **factory_kwargs,
#         )
#         self.act = nn.SiLU(inplace=True)
#
#         self.x_proj = (
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#         )
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
#         del self.x_proj
#
#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
#         del self.dt_projs
#
#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
#         self.Ds = self.D_init(self.d_inner, copies=2, merge=True)
#
#         self.selective_scan = selective_scan_fn
#         self.out_norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None
#         self.residual_weight = nn.Parameter(torch.tensor(0.1))
#
#
#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError
#
#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         dt_proj.bias._no_reinit = True
#         return dt_proj
#
#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log
#
#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=True):
#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)
#         D._no_weight_decay = True
#         return D
#
#     def forward_inpatch(self, x: torch.Tensor):
#         B, C, H, W = x.shape
#         L = H * W
#         K = 2
#         P = 2
#
#         y = x.transpose(2, 3).contiguous()
#         x = rearrange(x, 'b c (h p1) (w p2) -> b c (h w) (p1 p2)', p1=P, p2=P)
#         y = rearrange(y, 'b c (w p2) (h p1) -> b c (h w) (p2 p1)', p1=P, p2=P)
#
#         xs = torch.stack([x.view(B, -1, L), y.view(B, -1, L)], dim=1).view(B, 2, -1, L)
#         del x, y
#
#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         del x_dbl
#
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#
#         xs = xs.float().view(B, -1, L)
#         dts = dts.contiguous().float().view(B, -1, L)
#         Bs = Bs.float().view(B, K, -1, L)
#         Cs = Cs.float().view(B, K, -1, L)
#
#         Ds = self.Ds.float().view(-1)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1)
#
#         out_y = self.selective_scan(
#             xs, dts, As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)
#
#         del xs, dts, Bs, Cs, Ds, As, dt_projs_bias
#
#         y1 = out_y[:, 0].reshape(B, C, H // P, W // P, P, P).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
#
#         y2 = out_y[:, 1].reshape(B, C, H // P, W // P, P, P).permute(0, 1, 2, 4, 3, 5).reshape(B, C, W, H)
#
#         x_wh = y2.transpose(2, 3).contiguous()
#         out = y1 + x_wh
#         del x_wh
#         return out
#
#     def forward_patch(self, x: torch.Tensor):
#         B, C, H, W = x.shape
#         K = 2
#         P = 2
#         L = H // P * W // P
#
#         y = x.transpose(2, 3).contiguous()
#         x = rearrange(x, 'b c (h p1) (w p2) -> b c (h w) (p1 p2)', p1=P, p2=P).mean(dim=3)
#         y = rearrange(y, 'b c (w p2) (h p1) -> b c (h w) (p1 p2)', p1=P, p2=P).mean(dim=3)
#         xs = torch.stack([x, y], dim=1).view(B, 2, -1, L)
#         del x, y
#
#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         del x_dbl
#
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#
#         xs = xs.float().view(B, -1, L)
#         dts = dts.contiguous().float().view(B, -1, L)
#         Bs = Bs.float().view(B, K, -1, L)
#         Cs = Cs.float().view(B, K, -1, L)
#         Ds = self.Ds.float().view(-1)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1)
#
#         out_y = self.selective_scan(
#             xs, dts, As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)
#
#         del xs, dts, Bs, Cs, Ds, As, dt_projs_bias
#
#         wh_y = out_y[:, 1].view(B, -1, W // P, H // P).transpose(2, 3).contiguous().view(B, -1, L)
#
#         result = out_y[:, 0].add_(wh_y)
#         del out_y, wh_y
#
#         return result.view(B, -1, L, 1)
#
#     def forward_gobal(self, x: torch.Tensor):
#         B, C, H, W = x.shape
#         L = H * W
#         K = 2
#
#         xs = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#
#         xs = xs.float().view(B, -1, L)
#         dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
#         Bs = Bs.float().view(B, K, -1, L)
#         Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
#         Ds = self.Ds.float().view(-1)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
#
#         out_y = self.selective_scan(
#             xs, dts,
#             As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)
#         assert out_y.dtype == torch.float
#
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#
#         return out_y[:, 0] + wh_y
#
#
#     def forward(self, x: torch.Tensor, **kwargs):
#         B, H, W, C = x.shape
#         P = 2
#
#         xz = self.in_proj(x)
#         x, z = xz.chunk(2, dim=-1)
#
#         x = x.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)
#         x = self.act(self.conv2d(x))
#
#         y_inpatch = self.forward_inpatch(x)
#         y_inpatch = rearrange(y_inpatch, 'b c (h p1) (w p2) -> b c (h w) (p1 p2)', p1=P, p2=P)
#
#         y_patch = self.forward_patch(x)
#         y_gobal = torch.transpose(self.forward_gobal(x), dim0=1, dim1=2).contiguous().view(B, H, W, -1)
#         y_patch_gate = torch.sigmoid(y_patch)
#         y = y_patch_gate * y_inpatch
#         y = y.reshape(B, C * 2, H // P, W // P, P, P).permute(0, 1, 2, 4, 3, 5).reshape(B, 2*C, H, W)
#         y_local = y.permute(0, 2, 3, 1).contiguous()
#         y = y_local + y_gobal
#
#         y = self.out_norm(y)
#         y.mul_(F.silu(z))
#
#         out = self.out_proj(y)
#         if self.dropout is not None:
#             out = self.dropout(out)
#         return out
#
# class LFSSBlock(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             expand: float = 2.,
#             **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
#         self.drop_path = DropPath(drop_path)
#         self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
#         self.conv_blk = ffn(hidden_dim)
#         self.ln_2 = nn.LayerNorm(hidden_dim)
#         self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
#
#
#     def forward(self, input, x_size):
#         # x [B,HW,C]
#         B, L, C = input.shape
#         input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
#         x = self.ln_1(input)
#         x = input*self.skip_scale + self.drop_path(self.self_attention(x))
#         x = self.ln_2(x).permute(0, 3, 1, 2).contiguous()
#         x = input*self.skip_scale2 + self.conv_blk(x).permute(0, 2, 3, 1).contiguous()
#         x = x.view(B, -1, C).contiguous()
#         return x
#
#
#
# class LayerNormFunction(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, x, weight, bias, eps):
#         ctx.eps = eps
#         N, C, H, W = x.size()
#         mu = x.mean(1, keepdim=True)
#         var = (x - mu).pow(2).mean(1, keepdim=True)
#         y = (x - mu) / (var + eps).sqrt()
#         ctx.save_for_backward(y, var, weight)
#         y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
#         return y
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         eps = ctx.eps
#
#         N, C, H, W = grad_output.size()
#         y, var, weight = ctx.saved_variables
#         g = grad_output * weight.view(1, C, 1, 1)
#         mean_g = g.mean(dim=1, keepdim=True)
#
#         mean_gy = (g * y).mean(dim=1, keepdim=True)
#         gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
#         return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
#             dim=0), None
#
#
# class LayerNorm2d(nn.Module):
#
#     def __init__(self, channels, eps=1e-6):
#         super(LayerNorm2d, self).__init__()
#         self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
#         self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
#         self.eps = eps
#
#     def forward(self, x):
#         return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
#
#
#
#
# def batched_index_select(input, dim, index):
#     for ii in range(1, len(input.shape)):
#         if ii != dim:
#             index = index.unsqueeze(ii)
#     expanse = list(input.shape)
#     expanse[0] = -1
#     expanse[dim] = -1
#     index = index.expand(expanse)
#     return torch.gather(input, dim, index)
#
# def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
#     batch_size = input_maps.size(0) # B
#
#     if num_matches is None or num_matches == -1:
#         num_matches = input_maps.size(1)
#
#     topk_values, topk_indices = distances.topk(k=1, largest=False) # B, C, 1
#
#     topk_values = topk_values.squeeze(-1)
#     topk_indices = topk_indices.squeeze(-1)
#
#
#     sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
#     sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)
#
#     mask = torch.stack(
#         [
#             torch.where(sorted_indices_indices[i] < num_matches, True, False)
#             for i in range(batch_size)
#         ]
#     )
#
#     topk_indices_selected = topk_indices.masked_select(mask)
#     topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)
#     # indices = (
#     #     torch.arange(0, topk_values.size(1))
#     #     .unsqueeze(0)
#     #     .repeat(batch_size, 1)
#     #     .to(topk_values.device)
#     # )
#     # indices_selected = indices.masked_select(mask)
#     # indices_selected = indices_selected.reshape(batch_size, num_matches)
#     # filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
#     filtered_candidate_maps = batched_index_select(
#         candidate_maps, 1, topk_indices_selected
#     )
#
#     # return filtered_input_maps, filtered_candidate_maps
#     return filtered_candidate_maps
#
#
# def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
#     """
#     input_maps: (B, C, H*W)
#     candidate_maps: (B, C, H*W)
#     """
#
#     distances = torch.cdist(input_maps, candidate_maps) # B,C,C
#
#     return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)
#
# class Matching(nn.Module):
#     def __init__(self, dim=32, match_factor=1):
#         super(Matching, self).__init__()
#         self.num_matching = int(dim/match_factor)
#     def forward(self, x, perception):
#         b, c, h, w = x.size()
#         x = x.flatten(2, 3)
#         perception = perception.flatten(2, 3)
#         filtered_candidate_maps = neirest_neighbores_on_l2(x, perception, self.num_matching)
#         # filtered_input_maps = filtered_input_maps.reshape(b, self.num_matching, h, w)
#         filtered_candidate_maps = filtered_candidate_maps.reshape(b, self.num_matching, h, w)
#         return filtered_candidate_maps
#
#
# class PAConv(nn.Module):
#
#     def __init__(self, nf, k_size=3):
#         super(PAConv, self).__init__()
#         self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
#         self.sigmoid = nn.Sigmoid()
#         self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
#         self.k4 = nn.Conv2d(nf, nf//2, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
#
#     def forward(self, x):
#
#         y = self.k2(x)
#         y = self.sigmoid(y)
#
#         out = torch.mul(self.k3(x), y)
#         out = self.k4(out)
#
#         return out
#
# class Matching_transformation(nn.Module):
#     def __init__(self, dim=32, match_factor=1, ffn_expansion_factor=1, bias=True):
#         super(Matching_transformation, self).__init__()
#         self.num_matching = int(dim / match_factor)
#         self.channel = dim
#         hidden_features = int(self.channel * ffn_expansion_factor)
#         self.matching = Matching(dim=dim, match_factor=match_factor)
#         # self.matching = Matching(dim=dim)
#
#         self.paconv =  PAConv(dim*2)
#
#     def forward(self, x, perception):
#         filtered_candidate_maps = self.matching(x, perception)
#         # conv11 = self.conv11(concat)
#         concat = torch.cat([x, filtered_candidate_maps], dim=1)
#         out = self.paconv(concat)
#
#         return out
#
# class FeedForward(nn.Module):
#     def __init__(self, dim=32, match_factor=4, ffn_expansion_factor=1, bias=True, ffn_matching=True):
#         super(FeedForward, self).__init__()
#         self.num_matching = int(dim/match_factor)
#         self.channel = dim
#         self.matching = ffn_matching
#         hidden_features = int(self.channel * ffn_expansion_factor)
#
#         self.project_in = nn.Sequential(
#             nn.Conv2d(self.channel, hidden_features, 1, bias=bias),
#             nn.Conv2d(hidden_features, self.channel, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=bias)
#         )
#         if self.matching is True:
#             self.matching_transformation = Matching_transformation(dim=dim,
#                                                                    match_factor=match_factor,
#                                                                    ffn_expansion_factor=ffn_expansion_factor,
#                                                                    bias=bias)
#
#         self.project_out = nn.Sequential(
#             nn.Conv2d(self.channel, hidden_features, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=bias),
#             nn.GELU(),
#             nn.Conv2d(hidden_features, self.channel, 1, bias=bias))
#
#     def forward(self, x, perception):
#         project_in = self.project_in(x)
#         if perception is not None:
#             out = self.matching_transformation(project_in, perception)
#         else:
#             out = project_in
#         project_out = self.project_out(out)
#         return project_out
#
#
#
# ##########################################################################
# class CMTAttention(nn.Module):
#     def __init__(self, dim, num_heads, match_factor=4,ffn_expansion_factor=1,scale_factor=8, bias=True, attention_matching=True):
#         super(CMTAttention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.matching = attention_matching
#         if self.matching is True:
#             self.matching_transformation = Matching_transformation(dim=dim,
#                                                                    match_factor=match_factor,
#                                                                    ffn_expansion_factor=ffn_expansion_factor,
#                                                                    bias=bias)
#
#     def forward(self, x, perception):
#         b, c, h, w = x.shape
#
#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)
#
#         # perception = self.LayerNorm(perception)
#         if self.matching is True:
#             q = self.matching_transformation(q, perception)
#         else:
#             q = q
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         return out
#
#
# class FeedForward_Restormer(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor=1, bias=True):
#         super(FeedForward_Restormer, self).__init__()
#
#         hidden_features = int(dim * ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#
#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
#                                 groups=hidden_features * 2, bias=bias)
#
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x
#
#
#
# class ConvNeXtBlock(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
#
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#
#     def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(
#             dim, dim, kernel_size=3, padding=1
#         )  # depthwise conv
#         # self.norm = ConvNeXtBlockLayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(
#             dim, dim
#         )  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(dim, dim)
#         self.gamma = (
#             nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#             if layer_scale_init_value > 0
#             else None
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#
#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         # x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#         x = input + self.drop_path(x)
#         return x
#
#
#
#
# class DetailConv(nn.Module):
#     def __init__(
#             self,
#             dim,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#             depth_multiplier: int = 2,
#             drop_path: float = 0,
#     ):
#         super().__init__()
#         self.drop_path = DropPath(drop_path)
#         self.conv = nn.Conv2d(dim,dim,3,1,1)
#         self.Dconv = nn.Sequential(
#             nn.Conv2d(dim,dim*depth_multiplier,3,1,1, groups=dim),
#             nn.ReLU(dim),
#             nn.Conv2d(dim*depth_multiplier,dim,1),
#         )
#         self.relu = nn.ReLU()
#         self.ln_1 = norm_layer(dim)
#         self.skip_scale2 = nn.Parameter(torch.ones(1, dim, 1, 1))
#     def forward(self,input):
#         x = self.ln_1(input.permute(0, 2, 3, 1).contiguous())
#         x = self.Dconv(x.permute(0, 3, 1, 2).contiguous())
#         x = self.relu(x)
#         x_1 = self.conv(x)
#         x_out = input*self.skip_scale2 + x_1
#         return x_out
#
#
#
#
# class SDPBlock(nn.Module):
#     def __init__(
#             self,
#             dim,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(dim)
#         self.drop_path = DropPath(drop_path)
#         self.dconv = DetailConv(dim)
#
#         self.skip_scale= nn.Parameter(torch.ones(dim))
#         self.conv_hbg = ffn(dim)
#         self.ln_2 = nn.LayerNorm(dim)
#         self.skip_scale2 = nn.Parameter(torch.ones(dim))
#
#     def forward(self, input):
#         input = input.permute(0, 2, 3, 1).contiguous()
#         x = self.ln_1(input)
#
#         x = input*self.skip_scale + self.drop_path(self.dconv(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous())
#
#         x = x*self.skip_scale2 + self.conv_hbg(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
#         x= x.permute(0, 3, 1, 2).contiguous()
#         return x
#
#
# class HFEBlock(nn.Module):
#     def __init__(self,
#                  dim,
#                  n_d_blocks=2,
#     ):
#         super(HFEBlock, self).__init__()
#         self.l1_conv = nn.Conv2d(dim, dim, 3, 1, 1)
#         self.l_blk = nn.Sequential(*[SDPBlock(dim) for _ in range(n_d_blocks)])
#         self.skip_scale2 = nn.Parameter(torch.ones(1, dim, 1, 1))
#
#     def forward(self, x):
#         for l_layer in self.l_blk:
#             x = l_layer(x)
#
#         x_1 = self.l1_conv(x)+self.skip_scale2*x
#         return x_1
#
#
#
# class Resblock(nn.Module):
#     def __init__(self,dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(dim,dim,3,1,1)
#         self.conv2 = nn.Conv2d(dim,dim,3,1,1)
#         self.relu = nn.ReLU()
#         self.skip_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
#     def forward(self,input):
#         x = self.conv1(input)
#         x = self.relu(x)
#         x_out = self.skip_scale*input + self.conv2(x)
#         return x_out
#
# class ResGroup(nn.Module):
#     def __init__(self,dim,n_d_blocks=10):
#         super().__init__()
#         self.l_blk = nn.Sequential(*[Resblock(dim) for _ in range(n_d_blocks)])
#         self.skip_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
#     def forward(self,input):
#         x = input
#         for l_layer in self.l_blk:
#             x = l_layer(x)
#         x_out = self.skip_scale*input + x
#         return x_out
#
#
# class FullyAttentionalBlock(nn.Module):
#     def __init__(self, plane, norm_layer=nn.BatchNorm2d):
#         # 初始化函数，plane是输入和输出特征图的通道数，norm_layer是归一化层（默认为BatchNorm2d）
#         super(FullyAttentionalBlock, self).__init__()
#         # 定义两个全连接层，conv1和conv2
#         self.conv1 = nn.Linear(plane, plane)
#         self.conv2 = nn.Linear(plane, plane)
#
#         # 定义卷积层 + 归一化层 + 激活函数（ReLU）
#         self.conv = nn.Sequential(
#             nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),  # 卷积操作
#             norm_layer(plane),  # 归一化层
#             nn.ReLU()  # ReLU激活函数
#         )
#
#         # 定义softmax操作，用于计算关系矩阵
#         self.softmax = nn.Softmax(dim=-1)
#
#         # 初始化可学习的参数gamma，用于调整最终的输出
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         # 前向传播过程，x为输入的特征图，形状为 (batch_size, channels, height, width)
#         batch_size, _, height, width = x.size()
#
#         # 对输入张量进行排列和变形，获取水平和垂直方向的特征
#         feat_h = x.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)  # 水平方向特征
#         feat_w = x.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)  # 垂直方向特征
#
#         # 对输入张量分别在水平方向和垂直方向进行池化，并通过全连接层进行编码
#         encode_h = self.conv1(
#             F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())  # 水平方向编码
#         encode_w = self.conv2(
#             F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())  # 垂直方向编码
#
#         # 计算水平方向和垂直方向的关系矩阵
#         energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))  # 计算水平方向的关系
#         energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))  # 计算垂直方向的关系
#
#         # 计算经过softmax后的关系矩阵
#         full_relation_h = self.softmax(energy_h)  # 水平方向的关系
#         full_relation_w = self.softmax(energy_w)  # 垂直方向的关系
#
#         # 通过矩阵乘法和关系矩阵，对特征进行加权和增强
#         full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3,
#                                                                                                     1)  # 水平方向的增强
#         full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1,
#                                                                                                     3)  # 垂直方向的增强
#
#         # 将水平和垂直方向的增强特征进行融合，并加上原始输入特征
#         out = self.gamma * (full_aug_h + full_aug_w) + x
#
#         # 通过卷积层进行进一步的特征处理
#         out = self.conv(out)
#
#         return out  # 返回处理后的特征图
#
#
# class Fusion(nn.Module):
#     def __init__(self, dim, dropout_rate=0.5):  # 增加了 dropout_rate 参数
#         super().__init__()
#         # 定义一个1*1的卷积与中间处理层
#         self.conv1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
#         self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1)
#         self.relu = nn.ReLU()
#
#         # 添加 BatchNorm 层
#         self.bn1 = nn.BatchNorm2d(dim)  # BN 层用于 conv1
#         self.bn2 = nn.BatchNorm2d(dim)  # BN 层用于 conv3x3
#
#         # 其他卷积层
#         self.l_conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
#         self.l_conv2 = nn.Conv2d(dim, dim, 5, 1, 2)
#         self.resg = ResGroup(dim)
#         self.Fu_att = FullyAttentionalBlock(dim)
#         self.sigmoid = nn.Sigmoid()
#
#         # Dropout 层
#         self.dropout = nn.Dropout(dropout_rate)
#
#     def forward(self, spa, frq):
#         # 对 spa 和 frq 进行卷积并应用激活函数
#         x1 = self.relu(self.l_conv1(spa))
#         x2 = self.relu(self.l_conv2(frq))
#
#         # 将处理后的特征拼接
#         x_fu = torch.cat((x1, x2), dim=1)
#         x = self.conv1(x_fu)
#
#         # 应用 BatchNorm 和 ReLU
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         # 添加 Dropout 后续卷积层的输出
#         x = self.dropout(x)  # Dropout应用
#         x = self.resg(x)
#         x = self.conv3x3(x)
#
#         # 应用 BatchNorm 和激活函数
#         x = self.bn2(x)
#         x = self.relu(x)
#
#         # 特征增强
#         y = self.Fu_att(x)
#         wa = self.sigmoid(y)
#
#         # 融合最终输出
#         x_out = spa * wa + frq * (1 - wa)
#
#         return x_out
#
# class LocalGlobalAttention(nn.Module):
#     def __init__(self, output_dim, patch_size):
#         super().__init__()
#         self.output_dim = output_dim
#         self.patch_size = patch_size
#         self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
#         self.norm = nn.LayerNorm(output_dim // 2)
#         self.mlp2 = nn.Linear(output_dim // 2, output_dim)
#         self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
#         self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
#         self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         B, H, W, C = x.shape
#         P = self.patch_size
#
#         # Local branch
#         local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
#         local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
#         local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)
#
#         local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
#         local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
#         local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)
#
#         local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
#         local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)
#
#         cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
#         mask = cos_sim.clamp(0, 1)
#         local_out = local_out * mask
#         local_out = local_out @ self.top_down_transform
#
#         # Restore shapes
#         local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
#         local_out = local_out.permute(0, 3, 1, 2)
#         local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
#         output = self.conv(local_out)
#
#         return output
#
# class LAGB(nn.Module):
#     def __init__(self, dim, reduction=16):
#         super(LAGB, self).__init__()
#         # 用于提取低频图像特征
#         self.global_cap1 = LocalGlobalAttention(dim,patch_size = 2)
#         self.global_cap2 = LocalGlobalAttention(dim, patch_size=4)
#         self.dconv = nn.Conv2d(dim,dim,3,1,1)
#         self.relu = nn.ReLU(inplace=False)
#         # 使用FullyAttentionalBlock来生成通道注意力
#         self.channel_attention = FullyAttentionalBlock(dim)
#
#     def forward(self, low_freq, high_freq):
#         # 假设 high_freq 是一个张量，获取其形状
#         _, _, target_h, target_w = high_freq.shape  # 获取 high_freq 的高宽
#
#         # 动态调整 low_freq 的大小，使其与 high_freq 的形状一致
#         low_freq = F.interpolate(low_freq, size=(target_h, target_w), mode='bilinear', align_corners=False)
#
#         # 1. 提取低频和高频图像特征
#         low_feat1 = self.global_cap1(low_freq)
#         low_feat2 = self.global_cap2(low_freq)
#         low_feat3 = self.dconv(low_freq)
#         low_feat3 = self.relu(low_feat3)
#         low_feat_weight = (low_feat1+low_feat2+low_feat3)/3
#         # 3. 使用低频的池化特征生成通道注意力权重
#         attention_weights = self.channel_attention(low_feat_weight)  # 通过池化后的低频特征生成注意力权重
#
#         # 4. 应用通道注意力权重来加权高频特征
#         weighted_high = high_freq * attention_weights  # 加权高频图像特征
#
#         return weighted_high  # 返回加权后的高频特征
#
#
# #待改
# class DownSFG(nn.Module):
#     def __init__(self, dim, n_l_blocks=1, n_h_blocks=1 ,expand=2):
#         super().__init__()
#         self.fft = FFT(dim)
#         self.ifft = IFFT(dim)
#         self.fusion = Fusion(dim)
#         self.l_conv = nn.Conv2d(dim, dim, 5, 1, 2)
#         self.l_blk = nn.Sequential(*[LFSSBlock(dim, expand=expand) for _ in range(n_l_blocks)])
#         self.lag = LAGB(dim)
#         self.h_conv = nn.Conv2d(dim, dim, 5, 1, 2)
#         self.h_blk = nn.Sequential(*[HFEBlock(dim) for _ in range(n_h_blocks)])
#
#     def forward(self, x, x_space):
#         x_HH, x_LL = self.fft(x)
#         b, c, h, w = x_LL.shape
#         x_LL = self.fusion(x_LL,x_space)
#         x_LL = self.l_conv(x_LL)
#         x_LL = rearrange(x_LL, "b c h w -> b (h w) c").contiguous()
#         for l_layer in self.l_blk:
#             x_LL = l_layer(x_LL, [h, w])
#         x_LL = rearrange(x_LL, "b (h w) c -> b c h w", h=h, w=w).contiguous()
#         x_HH = self.lag(x_LL,x_HH)
#         x_h = self.h_conv(x_HH)
#         for h_layer in self.h_blk:
#             x_h = h_layer(x_h)
#         x_fre = self.ifft(x_LL, x_h)
#         return x_fre
#
#
# #待改
# class upSFG(nn.Module):
#     def __init__(self,dim, n_l_blocks=1, n_h_blocks=1 ,expand=2):
#         super().__init__()
#         self.fft = FFT(dim)
#         self.ifft = IFFT(dim)
#         self.l_conv = nn.Conv2d(dim, dim, 3, 1, 1)
#         self.l_blk = nn.Sequential(*[LFSSBlock(dim, expand=expand) for _ in range(n_l_blocks)])
#         self.lag = LAGB(dim)
#         self.h_conv = nn.Conv2d(dim, dim, 3, 1, 1)
#         self.h_blk = nn.Sequential(*[HFEBlock(dim) for _ in range(n_h_blocks)])
#         self.conv_l = nn.Conv2d(dim*2,dim,3,1,1)
#
#     def forward(self, x, x_space):
#         x = torch.cat((x, x_space), dim=1)
#         x = self.conv_l(x)
#         x_HH, x_LL = self.fft(x)
#         b, c, h, w = x_LL.shape
#         x_LL = self.l_conv(x_LL)
#
#         x_LL = rearrange(x_LL, "b c h w -> b (h w) c").contiguous()
#         for l_layer in self.l_blk:
#             x_LL = l_layer(x_LL, [h, w])
#         x_LL = rearrange(x_LL, "b (h w) c -> b c h w", h=h, w=w).contiguous()
#         x_HH = self.lag(x_LL, x_HH)
#         x_h = self.h_conv(x_HH)
#         for h_layer in self.h_blk:
#             x_h = h_layer(x_h)
#         x_fre = self.ifft(x_LL, x_h)
#         return x_fre
#
# class down_sample(nn.Module):
#     def __init__(self,dim):
#         super().__init__()
#         self.ps_down = nn.Sequential(
#             nn.PixelUnshuffle(2),
#             nn.Conv2d((2 ** 2) * dim, 2 * dim, kernel_size=3, stride=1, padding=1, bias=False),
#         )
#     def forward(self,x):
#         x = self.ps_down(x)
#         return x
#
# class up_sample(nn.Module):
#     def __init__(self,dim):
#         super().__init__()
#         self.ps_up = nn.Sequential(
#             nn.Conv2d( 2 * dim, (2 ** 2) * dim, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.PixelShuffle(2)
#         )
#     def forward(self,x):
#         x = self.ps_up(x)
#         return x
#
# class UNet(nn.Module):
#     def __init__(self, in_chn=3, wf=16, n_l_blocks=[1,2,4], n_h_blocks=[1,2,2], ffn_scale=2):
#         super(UNet, self).__init__()
#         self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
#         self.ps_down = nn.Sequential(
#             nn.PixelUnshuffle(2),
#             nn.Conv2d((2 ** 2) * wf, wf, kernel_size=3, stride=1, padding=1, bias=False),
#         )
#         self.ps_down1 = down_sample(wf)
#         self.ps_down2 = down_sample(wf*2)
#         self.ps_down3 = down_sample(wf*4)
#
#         self.ps_up1 = up_sample(wf*4)
#         self.ps_up2 = up_sample(wf*2)
#         self.ps_up3 = up_sample(wf)
#
#
#         # encoder of UNet-64
#         self.group1 = DownSFG(wf, n_l_blocks=n_l_blocks[0], n_h_blocks=n_h_blocks[0], expand=ffn_scale)
#         self.group2 = DownSFG(wf*2, n_l_blocks=n_l_blocks[1], n_h_blocks=n_h_blocks[1], expand=ffn_scale)
#         self.group3 = DownSFG(wf*4, n_l_blocks=n_l_blocks[2], n_h_blocks=n_h_blocks[2], expand=ffn_scale)
#         self.group4 = DownSFG(wf*8, n_l_blocks=4, n_h_blocks=2, expand=ffn_scale)
#
#         # decoder of UNet-64
#         self.up_group3 = upSFG(wf*4, n_l_blocks=n_l_blocks[2], n_h_blocks=n_h_blocks[2], expand=ffn_scale)
#         self.up_group2 = upSFG(wf*2, n_l_blocks=n_l_blocks[1], n_h_blocks=n_h_blocks[1], expand=ffn_scale)
#         self.up_group1 = upSFG(wf, n_l_blocks=n_l_blocks[0], n_h_blocks=n_h_blocks[0], expand=ffn_scale)
#
#         self.last = nn.Conv2d(wf, in_chn, kernel_size=3, stride=1, padding=1, bias=True)
#
#     def forward(self, x):
#         img = x
#         ##### shallow conv #####
#         x_shallow = self.conv_01(img)
#         img_down1 = self.ps_down(x_shallow)
#         img_down2 = self.ps_down1(img_down1)
#         img_down3 = self.ps_down2(img_down2)
#         img_down4 = self.ps_down3(img_down3)
#         ######## UNet-64 ########
#         # Down-path (Encoder)
#         x_1 = self.group1(x_shallow,img_down1)
#         x_down1 = self.ps_down1(x_1)
#         x_2 = self.group2(x_down1,img_down2)
#         x_down2 = self.ps_down2(x_2)
#         x_3 = self.group3(x_down2,img_down3)
#         x_down3 = self.ps_down3(x_3)
#
#         x_4 = self.group4(x_down3,img_down4)
#
#         # Up-path (Decoder)
#         x_up1 = self.ps_up1(x_4)
#         x_5 = self.up_group3(x_up1, x_3)
#         x_up2 = self.ps_up2(x_5)
#         x_6 = self.up_group2(x_up2, x_2)
#         x_up3 = self.ps_up3(x_6)
#         x_free = self.up_group1(x_up3, x_1)
#         ##### Reconstruct #####
#         out_1 = self.last(x_free) + img
#         return out_1
#
#
# @ARCH_REGISTRY.register()
# class SFMamba(nn.Module):
#     def __init__(self,
#                  *,
#                  in_chn,
#                  wf,
#                  n_l_blocks=[1,1,2],
#                  n_h_blocks=[1,1,1],
#                  ffn_scale=2.0,
#                  **ignore_kwargs):
#         super().__init__()
#         self.restoration_network = UNet(in_chn=in_chn, wf=wf, n_l_blocks=n_l_blocks, n_h_blocks=n_h_blocks, ffn_scale=ffn_scale)
#
#     def print_network(self, model):
#         num_params = 0
#         for p in model.parameters():
#             num_params += p.numel()
#         print(model)
#         print("The number of parameters: {}".format(num_params))
#
#     def encode_and_decode(self, input, current_iter=None):
#
#         restoration = self.restoration_network(input)
#         return restoration
#
#     @torch.no_grad()
#     def test_tile(self, input, tile_size=240, tile_pad=16):
#         # return self.test(input)
#         """It will first crop input images to tiles, and then process each tile.
#         Finally, all the processed tiles are merged into one images.
#         Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
#         """
#         batch, channel, height, width = input.shape
#         output_height = height * self.scale_factor
#         output_width = width * self.scale_factor
#         output_shape = (batch, channel, output_height, output_width)
#
#         # start with black image
#         output = input.new_zeros(output_shape)
#         tiles_x = math.ceil(width / tile_size)
#         tiles_y = math.ceil(height / tile_size)
#
#         # loop over all tiles
#         for y in range(tiles_y):
#             for x in range(tiles_x):
#                 # extract tile from input image
#                 ofs_x = x * tile_size
#                 ofs_y = y * tile_size
#                 # input tile area on total image
#                 input_start_x = ofs_x
#                 input_end_x = min(ofs_x + tile_size, width)
#                 input_start_y = ofs_y
#                 input_end_y = min(ofs_y + tile_size, height)
#
#                 # input tile area on total image with padding
#                 input_start_x_pad = max(input_start_x - tile_pad, 0)
#                 input_end_x_pad = min(input_end_x + tile_pad, width)
#                 input_start_y_pad = max(input_start_y - tile_pad, 0)
#                 input_end_y_pad = min(input_end_y + tile_pad, height)
#
#                 # input tile dimensions
#                 input_tile_width = input_end_x - input_start_x
#                 input_tile_height = input_end_y - input_start_y
#                 tile_idx = y * tiles_x + x + 1
#                 input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
#
#                 # upscale tile
#                 output_tile = self.test(input_tile)
#
#                 # output tile area on total image
#                 output_start_x = input_start_x * self.scale_factor
#                 output_end_x = input_end_x * self.scale_factor
#                 output_start_y = input_start_y * self.scale_factor
#                 output_end_y = input_end_y * self.scale_factor
#
#                 # output tile area without padding
#                 output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
#                 output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
#                 output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
#                 output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor
#
#                 # put tile into output image
#                 output[:, :, output_start_y:output_end_y,
#                 output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
#                                                output_start_x_tile:output_end_x_tile]
#         return output
#
#     def check_image_size(self, x, window_size=8):
#         _, _, h, w = x.size()
#         mod_pad_h = (window_size - h % (window_size)) % (
#             window_size)
#         mod_pad_w = (window_size - w % (window_size)) % (
#             window_size)
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#         return x
#
#     @torch.no_grad()
#     def test(self, input):
#         _, _, h_old, w_old = input.shape
#
#         restoration = self.encode_and_decode(input)
#
#         output = restoration
#
#         return output
#
#     def forward(self, input):
#
#         restoration = self.encode_and_decode(input)
#
#         return restoration
#
#
# if __name__== '__main__':
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(1, 3, 1920, 1280).to(device)
#     model = UNet(in_chn=3, wf=16, n_l_blocks=[1,2,4], n_h_blocks=[1,1,1], ffn_scale=2).to(device)
# #    print(model)
#     inp_shape=(3,512, 512)
#     from ptflops import get_model_complexity_info
#     FLOPS = 0
#     macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=True)#打印参数复杂度
#
#     params = float(params[:-4])
#     print('mac', macs)
#     print(params)
#     macs = float(macs[:-4]) + FLOPS / 10 ** 9
#
#     print('mac', macs)
#     print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
#     # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
#     with torch.no_grad():
#         torch.cuda.reset_max_memory_allocated(device)
#         start_time = time.time()
#         output = model(x)
#         end_time = time.time()
#         memory_used = torch.cuda.max_memory_allocated(device)
#     running_time = end_time - start_time
#     print(output.shape)
#     print(running_time)
#     print(f"Memory used: {memory_used / 1024**3:.3f} GB")


import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from PIL.ImageOps import expand
from scipy.io import savemat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
from typing import Optional, Callable
import math
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys

from torch.fft import ifft2
from torch.nn.functional import relu_

from basicsr.utils.registry import ARCH_REGISTRY
import torch.autograd
import numpy as np
import torch.fft as fft



def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

def iwt_weight_expand(weight):
    """
    weight: (B, 4E, H/2, W/2)
    return: (B, E, H, W)
    """
    B, CE, H, W = weight.shape
    E = CE // 4

    w = weight.view(B, 4, E, H, W)  # (B,4,E,H/2,W/2)

    x1 = w[:, 0] / 2  # HL
    x2 = w[:, 1] / 2  # LH
    x3 = w[:, 2] / 2  # HH
    x4 = w[:, 3] / 2  # LL

    out = torch.zeros(B, E, H*2, W*2, device=weight.device)

    out[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    out[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    out[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    out[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return out

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.relu = nn.GELU()

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)



class HazeAwareMoE(nn.Module):
    def __init__(self, dim, num_experts=4, topk=2):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.DWT = DWT()
        self.gate_feat = nn.Linear(dim, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(num_experts)
        ])

        self.conv = nn.Conv2d(dim, num_experts, 3, 1, 1)

    def forward(self, x):

        LL, LH, HL, HH = self.DWT(x)

        e_LL = self.conv(LL)
        e_LH = self.conv(LH)
        e_HL = self.conv(HL)
        e_HH = self.conv(HH)

        energy_map = torch.cat([e_HL, e_LH, e_HH, e_LL], dim=1)
        energy_map = iwt_weight_expand(energy_map)
        # energy_map = self.conv(x)
        B, C, H, W = x.shape

        x_flat = rearrange(x, 'b c h w -> (b h w) c')

        logits_feat = self.gate_feat(x_flat)
        logits_haze = rearrange(energy_map, 'b c h w -> (b h w) c')

        logits = logits_feat + self.alpha * logits_haze

        # ===== 概率 =====
        probs = F.softmax(logits, dim=1)   # (N, E)

        # ===== Top-k =====
        topk_val, topk_idx = torch.topk(logits, self.topk, dim=1)
        gates = F.softmax(topk_val, dim=1)

        # ===== importance =====
        importance = probs.sum(0)  # (E,)

        # ===== load =====
        load = torch.zeros(self.num_experts, device=x.device)

        for i in range(self.num_experts):
            load[i] = (topk_idx == i).sum()

        # ===== normalize =====
        importance = importance / importance.sum()
        load = load / load.sum()

        # ===== balance loss =====
        loss_balance = (importance * load).sum() * self.num_experts

        # ===== dispatch =====
        out = torch.zeros_like(x_flat)

        for expert_id in range(self.num_experts):
            mask = (topk_idx == expert_id)
            if not mask.any():
                continue

            token_idx, which_k = torch.where(mask)
            gate_val = gates[token_idx, which_k].unsqueeze(1)

            expert_inp = x_flat[token_idx]
            expert_out = self.experts[expert_id](expert_inp)

            out[token_idx] += gate_val * expert_out

        out = rearrange(out, '(b h w) c -> b c h w', b=B, h=H, w=W)

        return out, loss_balance



class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU(inplace=True)

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)

        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.weight_local = nn.Parameter(torch.ones(1, 1, self.d_model*2))
        self.weight_global = nn.Parameter(torch.ones(1, 1, self.d_model*2))



    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_inpatch(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 2
        P = 2

        y = x.transpose(2, 3).contiguous()
        x = rearrange(x, 'b c (h p1) (w p2) -> b c (h w) (p1 p2)', p1=P, p2=P)
        y = rearrange(y, 'b c (w p2) (h p1) -> b c (h w) (p2 p1)', p1=P, p2=P)

        xs = torch.stack([x.view(B, -1, L), y.view(B, -1, L)], dim=1).view(B, 2, -1, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        del x_dbl

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        out_path1 = rearrange(out_y[:, 0].reshape(B, C, H // P, W // P, P, P),
                              'b c h w p1 p2 -> b c (h p1) (w p2)')  # (B, C, H, W)

        # 路径2：转置图像，还原到转置空间 (W, H) 再转置回来
        out_path2_t = rearrange(out_y[:, 1].reshape(B, C, H // P, W // P, P, P),
                                'b c h w p2 p1 -> b c (w p2) (h p1)')  # (B, C, W, H)
        out_path2 = out_path2_t.transpose(2, 3)  # (B, C, H, W)

        out = out_path1 + out_path2
        return out


    def forward_patch(self, x: torch.Tensor):
        B, C, H, W = x.shape
        K = 2
        P = 2
        L = H // P * W // P

        y = x.transpose(2, 3).contiguous()
        x = rearrange(x, 'b c (h p1) (w p2) -> b c (h w) (p1 p2)', p1=P, p2=P).mean(dim=3)
        y = rearrange(y, 'b c (w p2) (h p1) -> b c (h w) (p1 p2)', p1=P, p2=P).mean(dim=3)
        xs = torch.stack([x, y], dim=1).view(B, 2, -1, L)
        del x, y

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        del x_dbl

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        del xs, dts, Bs, Cs, Ds, As, dt_projs_bias

        wh_y = out_y[:, 1].view(B, -1, W // P, H // P).transpose(2, 3).contiguous().view(B, -1, L)

        result = out_y[:, 0].add_(wh_y)
        del out_y, wh_y
        return result.view(B, -1, L, 1)

    def forward_gobal(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 2

        xs = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0] + wh_y


    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        P = 2

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)
        x = self.act(self.conv2d(x))

        y_inpatch = self.forward_inpatch(x)
        y_inpatch = rearrange(y_inpatch, 'b c (h p1) (w p2) -> b c (h w) (p1 p2)', p1=P, p2=P)

        y_patch = self.forward_patch(x)
        y_global = torch.transpose(self.forward_gobal(x), dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_patch_gate = torch.sigmoid(y_patch)
        y = y_patch_gate * y_inpatch
        y = y.reshape(B, C * 2, H // P, W // P, P, P).permute(0, 1, 2, 4, 3, 5).reshape(B, 2*C, H, W)
        y_local = y.permute(0, 2, 3, 1).contiguous()
        y = y_local + y_global

        y = self.out_norm(y)
        y.mul_(F.silu(z))
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class GMambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = HazeAwareMoE(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = self.ln_2(x).permute(0, 3, 1, 2).contiguous()
        x, loss_balance = self.conv_blk(x)
        x = input*self.skip_scale2 + x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x, loss_balance



class SFG(nn.Module):
    def __init__(self, dim,  n_h_blocks=1 ,expand=2):
        super().__init__()
        self.h_conv = nn.Conv2d(dim, dim, 5, 1, 2)
        self.h_blk = nn.Sequential(*[GMambaBlock(dim, expand=expand) for _ in range(n_h_blocks)])

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.h_conv(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()

        total_loss = 0

        for h_layer in self.h_blk:
            x, loss = h_layer(x, [h, w])
            total_loss += loss

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        return x, total_loss


#downsample
class down_sample(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.ps_down = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d((2 ** 2) * dim, 2 * dim, kernel_size=3, stride=1, padding=1, bias=False),
        )
    def forward(self,x):
        x = self.ps_down(x)
        return x

#upsample
class up_sample(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.ps_up = nn.Sequential(
            nn.Conv2d( 2 * dim, (2 ** 2) * dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self,x):
        x = self.ps_up(x)
        return x




class UNet(nn.Module):
    def __init__(self, in_chn=3, wf=32,  n_h_blocks=[1,1,1], ffn_scale=2):
        super(UNet, self).__init__()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(wf*8, wf*4, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(wf*4, wf*2, 3, 1, 1)
        self.conv_up3 = nn.Conv2d(wf*2, wf, 3, 1, 1)
        self.ps_down1 = down_sample(wf)
        self.ps_down2 = down_sample(wf*2)
        self.ps_down3 = down_sample(wf*4)

        self.ps_up1 = up_sample(wf*4)
        self.ps_up2 = up_sample(wf*2)
        self.ps_up3 = up_sample(wf)


        # encoder of UNet-64
        self.group1 = SFG(wf, n_h_blocks=n_h_blocks[0], expand=ffn_scale)
        self.group2 = SFG(wf*2,  n_h_blocks=n_h_blocks[1], expand=ffn_scale)
        self.group3 = SFG(wf*4, n_h_blocks=n_h_blocks[2], expand=ffn_scale)

        self.group4 = SFG(wf*8,  n_h_blocks=4, expand=ffn_scale)

        # decoder of UNet-64
        self.up_group3 = SFG(wf*4,  n_h_blocks=n_h_blocks[2], expand=ffn_scale)
        self.up_group2 = SFG(wf*2,  n_h_blocks=n_h_blocks[1], expand=ffn_scale)
        self.up_group1 = SFG(wf,  n_h_blocks=n_h_blocks[0], expand=ffn_scale)

        self.last = nn.Conv2d(wf, in_chn, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        img = x

        ##### shallow conv #####
        x_shallow = self.conv_01(img)
        ######## UNet-64 ########
        # Down-path (Encoder)
        x_1, loss1= self.group1(x_shallow)
        x_down1 = self.ps_down1(x_1)
        x_2, loss2 = self.group2(x_down1)
        x_down2 = self.ps_down2(x_2)
        x_3, loss3 = self.group3(x_down2)
        x_down3 = self.ps_down3(x_3)
        x_4, loss4 = self.group4(x_down3)

        x_up1 = self.ps_up1(x_4)
        x4 = torch.cat([x_3, x_up1], dim=1)
        x4 = self.conv_up1(x4)
        x_5, loss5 = self.up_group3(x4)

        x_up2 = self.ps_up2(x_5)
        x5 = torch.cat([x_2, x_up2], dim=1)
        x5 = self.conv_up2(x5)
        x_6, loss6 = self.up_group2(x5)

        x_up3 = self.ps_up3(x_6)
        x6 = torch.cat([x_1, x_up3], dim=1)
        x6 = self.conv_up3(x6)
        x_free, loss7 = self.up_group1(x6)
        total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
        out_1 = self.last(x_free) + img
        return out_1, total_loss

@ARCH_REGISTRY.register()
class SFMamba(nn.Module):
    def __init__(self,
                 *,
                 in_chn,
                 wf,
                 n_h_blocks=[1,1,1],
                 ffn_scale=2.0,
                 **ignore_kwargs):
        super().__init__()
        self.restoration_network = UNet(in_chn=in_chn, wf=wf,  n_h_blocks=n_h_blocks, ffn_scale=ffn_scale)

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def encode_and_decode(self, input, current_iter=None):

        restoration = self.restoration_network(input)
        return restoration

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output

    def check_image_size(self, x, window_size=8):
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % (window_size)) % (
            window_size)
        mod_pad_w = (window_size - w % (window_size)) % (
            window_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    @torch.no_grad()
    def test(self, input):
        _, _, h_old, w_old = input.shape

        restoration = self.encode_and_decode(input)

        output = restoration

        return output

    def forward(self, input):

        restoration = self.encode_and_decode(input)

        return restoration


if __name__== '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 1920, 1280).to(device)
    model = UNet(in_chn=3, wf=16, n_h_blocks=[1,1,1], ffn_scale=2).to(device)
    inp_shape=(3,512, 512)
    from ptflops import get_model_complexity_info
    FLOPS = 0
    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=True)

    params = float(params[:-4])
    print('mac', macs)
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated(device)
        start_time = time.time()
        output = model(x)
        end_time = time.time()
        memory_used = torch.cuda.max_memory_allocated(device)
    running_time = end_time - start_time
    print(output.shape)
    print(running_time)
    print(f"Memory used: {memory_used / 1024**3:.3f} GB")
