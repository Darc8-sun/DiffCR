import torch
import torch.nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import math
from ldm.modules.diffusionmodules.util import (conv_nd,
    linear,
    normalization,timestep_embedding
)
from transformers.activations import quick_gelu
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
import torch.fft as fft
from ldm.modules.attention import BasicTransformerBlock,rearrange
from torch import Tensor

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std
def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor,bits:int):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    style_mean, style_std = bitquant(style_mean,bits=bits),bitquant(style_std,bits=bits)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def bitquant(tensor,bits=8):
    return torch.round(tensor*(2**bits-1)+0.5)/(2**bits-1)


class my_crossattention(nn.Module):
    def __init__(self, inner_dim, n_heads, d_head,dropout=0.,disable_self_attn=False,use_checkpoint=True):
        super().__init__()
        assert inner_dim == n_heads * d_head
        # self.norm = Normalize(in_channels)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=inner_dim,
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)]
        )

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        # x = self.norm(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        context[0]= rearrange(context[0], 'b c h w -> b (h w) c').contiguous()
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        return x






def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)



def conv1x1(in_ch: int, out_ch: int, stride: int = 1,bias=True) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride,bias=bias)




def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)




class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)





class Time_aware_FQCA(nn.Module):
    def __init__(self, in_channels,inner_dim,out_dim, n_heads, d_head):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.conv_in_c = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.cross_attetion_h= my_crossattention(inner_dim,n_heads,d_head,disable_self_attn=False,use_checkpoint=False)
        self.cross_attetion_l = my_crossattention(inner_dim, n_heads, d_head, disable_self_attn=False,use_checkpoint=False)
        self.mask_low=None
        self.mask_high = None
        self.out= nn.Sequential(
            conv_nd(2, inner_dim, out_dim, 3, padding=1),
            normalization(out_dim),
            nn.SiLU(),
        )
    def latent_2_fourier(self,x,threshold_ratio=1.0,ot=False):
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))
        if ot:
            return x_freq
        B, C, H, W = x_freq.shape
        if self.mask_high==None or x_freq.shape!=self.mask_high.shape:
            self.mask_high = torch.ones((B, C, H, W)).to(x.device)
            crow, ccol = H // 2, W // 2
            threshold_row = int(crow * threshold_ratio // 4)
            threshold_col = int(ccol * threshold_ratio // 4)
            self.mask_high[..., crow - threshold_row:crow + threshold_row, ccol - threshold_col:ccol + threshold_col] = 0
            self.mask_low = 1.0 - self.mask_high
        x_freq_high = x_freq * self.mask_high
        x_freq_low = x_freq * self.mask_low
        return x_freq_high,x_freq_low
    def fourier_2_latent(self,x):
        x = fft.ifftshift(x, dim=(-2, -1))
        x = fft.ifftn(x, dim=(-2, -1)).real
        return x
    def forward(self,x,content,t):
        coef_t=t.view(-1,1,1,1)/1000.0
        x=self.conv_in(x)
        content=self.conv_in_c(content)
        x_h,x_l=self.latent_2_fourier(x,1.0)
        content_h, content_l = self.latent_2_fourier(content, 1.0)
        x_h_2=self.cross_attetion_h(self.fourier_2_latent(x_h),context=self.fourier_2_latent(content_h))
        x_h=self.latent_2_fourier(x_h_2,ot=True)*self.mask_high*(1.0-coef_t)+x_h*coef_t


        x_l_2 = self.cross_attetion_l(self.fourier_2_latent(x_l), context=self.fourier_2_latent(content_l))
        x_l  = self.latent_2_fourier(x_l_2,ot=True)*self.mask_low*coef_t+x_l*(1.0-coef_t)
        x = fft.ifftshift(x_h+x_l, dim=(-2, -1))
        x = fft.ifftn(x, dim=(-2, -1)).real
        return self.out(x)

class Time_aware_refinementX(nn.Module):
    def __init__(self, in_ch,out_ch,model_channels,num_grow_ch=32,use_oai=False):
        super().__init__()
        # self.in_layer=nn.Sequential(
        #     conv_nd(2, in_ch, in_ch, 3, padding=1),
        #     nn.SiLU(),
        # )
        self.model_channels=model_channels
        self.FQCA=Time_aware_FQCA(in_ch,96,out_dim=32,d_head=32,n_heads=3)
        self.time_embed = nn.Sequential(
            linear(model_channels, model_channels*4),
            nn.SiLU(),
            linear(model_channels*4, model_channels*4),
        )
        self.in_layer=nn.Sequential(
            conv_nd(2, 4, 32, 3, padding=1),
            normalization(32),
            nn.SiLU(),
        )
        self.use_oai=use_oai
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            linear(
                model_channels * 4,
                in_ch, ), )
        self.mid_layer=make_layer(RRDB, 24, num_feat=64, num_grow_ch=num_grow_ch)
        # self.mid_layer = ResidualDenseBlock(num_feat=in_ch * 2, num_grow_ch=num_grow_ch)
        self.out_layer=nn.Sequential(
            conv_nd(2, 64, 64, 3, padding=1),
            normalization(64),
            nn.SiLU(),
            conv_nd(2, 64, out_ch, 3, padding=1),
        )

    def forward(self,h,c_c,t,res=None):
        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        t_emb = self.emb_layer(self.time_embed(t_emb)).type(h.dtype)
        while len(t_emb.shape) < len(h.shape):
            t_emb = t_emb[..., None]
        c=self.FQCA(c_c,h,t)
        h = h + t_emb
        h=self.in_layer(h)
        h = torch.cat([h, c], dim=1)
        h=self.out_layer(self.mid_layer(h))

        return h+c_c



class ProjLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)

        self.LayerNorm = nn.LayerNorm(in_dim, eps=eps)

    def forward(self, x):
        x = self.LayerNorm(x)
        x = self.dropout(self.dense2(quick_gelu((self.dense1(x)))))

        return x