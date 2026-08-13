import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
import math
from einops import rearrange
bn_mom = 0.1
BatchNorm2d = nn.BatchNorm2d



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            # nn.ReLU6()
            nn.ReLU()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class RemoteLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True,
                 drop_path=0.
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)

        self.dwconv7 = nn.Conv2d(dim*2, dim*2, kernel_size=7, padding=3, groups=dim*2)
        self.dwconv5 = nn.Conv2d(dim*2, dim*2, kernel_size=5, padding=2, groups=dim*2)
        self.dwconv3 = nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1, groups=dim*2)

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.pwconv1 = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.act = nn.GELU()

        self.pwconv2 = nn.Conv2d(2*dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.localproj = nn.Conv2d(dim, dim, 1)

        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.interconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()

        if W % ps != 0 and H % ps != 0:
            x = F.pad(x, (0, ps - W % ps, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        # local
        input = self.act(self.pwconv1(self.norm(x)))

        local7 = self.dwconv7(input)

        local5 = self.dwconv5(input)

        local3 = self.dwconv3(input)

        localall = local7+local3+local5

        local_branch0 = self.pwconv2(localall)
        local_branch1 = self.conv1(x)
        local = local_branch0 * local_branch1
        local = self.localproj(local)
        local = x + self.drop_path(local)


        # remote
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.interconv(attn)

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class RLB_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# remote local block
class RLB(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RemoteLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = RLB_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def Make_RLBlock(inplanes, num_blocks):

    layers = []
    layers.append(RLB(dim=inplanes, num_heads=4, window_size=8))
    for i in range(1, num_blocks):
        if i == (num_blocks - 1):
            layers.append(RLB(inplanes, num_heads=4, window_size=8))
        else:
            layers.append(RLB(inplanes, num_heads=8, window_size=8))

    return nn.Sequential(*layers)



class FeatureFusionModule(nn.Module):
    """
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_chan=256, out_chan=256, k_size=5):
        super(FeatureFusionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        #
        self.convblk = ConvBNReLU(in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_chan)
    def forward(self, x1, x2):

        x = torch.cat((x1, x2), dim=1)
        x = self.convblk(x)
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.bn(y)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=8):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear',align_corners=False)

        return out


class BAFNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=6, embed_dims=[32, 64, 160, 256],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 5, 2], num_stages=4):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.relu = nn.ReLU(inplace=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0


        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                               embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size//4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                               embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                               embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                               embed_dim=embed_dims[3])
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], mlp_ratio=mlp_ratios[0], drop=drop_rate, drop_path=dpr[cur + j])
            for j in range(depths[0])])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], mlp_ratio=mlp_ratios[1], drop=drop_rate, drop_path=dpr[cur + j])
            for j in range(depths[1])])
        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], mlp_ratio=mlp_ratios[2], drop=drop_rate, drop_path=dpr[cur + j])
            for j in range(depths[2])])
        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], mlp_ratio=mlp_ratios[3], drop=drop_rate, drop_path=dpr[cur + j])
            for j in range(depths[3])])

        self.norm1 = norm_layer(embed_dims[0])
        self.norm2 = norm_layer(embed_dims[1])
        self.norm3 = norm_layer(embed_dims[2])
        self.norm4 = norm_layer(embed_dims[3])

        self.det_planes = 128   #detail channels


        self.compression3 = nn.Sequential(
            nn.Conv2d(embed_dims[2], self.det_planes, kernel_size=1, bias=False),
            BatchNorm2d(self.det_planes, momentum=bn_mom),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(embed_dims[3], self.det_planes, kernel_size=1, bias=False),
            BatchNorm2d(self.det_planes, momentum=bn_mom),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(self.det_planes, embed_dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(embed_dims[2], momentum=bn_mom),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(self.det_planes, embed_dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(embed_dims[2], momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(embed_dims[3], momentum=bn_mom),
        )

        self.layer3_ = Make_RLBlock(self.det_planes, 2)
        self.layer4_ = Make_RLBlock(self.det_planes, 1)
        self.layer5_ = Make_RLBlock(self.det_planes, 1)

        self.convk1_det = nn.Sequential(
            nn.Conv2d(embed_dims[1], self.det_planes, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.det_planes, momentum=bn_mom)
        )


        self.convk1_sem = nn.Sequential(
            nn.Conv2d(embed_dims[3], self.det_planes, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.det_planes, momentum=bn_mom)
        )

        self.FFM = FeatureFusionModule(k_size=5)



        self.final_layer = segmenthead(self.det_planes*2, self.det_planes, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)


    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        B = x.shape[0]
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        layers.append(x)


        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        layers.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        layers.append(x)

        # first interact

        x_ = self.layer3_(self.relu(self.convk1_det(layers[1])))

        x = x + self.down3(self.relu(x_))

        x_ = x_ + F.interpolate(
            self.compression3(self.relu(layers[2])),
            size=[height_output, width_output],
            mode='bilinear',align_corners=False)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        layers.append(x)

        # second interact
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))

        x_ = x_ + F.interpolate(
            self.compression4(self.relu(layers[3])),
            size=[height_output, width_output],
            mode='bilinear',align_corners=False)

        x_ = self.layer5_(self.relu(x_))

        x = F.interpolate(
            self.convk1_sem(self.relu(x)),
            size=[height_output, width_output],
            mode='bilinear',align_corners=False)

        x_cat = self.FFM(x, x_)
        x_cat = self.final_layer(x_cat)


        return x_cat




def build_model(pretrained=True, **kwargs):
    model = BAFNet(num_classes=6, embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = "https://huggingface.co/Visual-Attention-Network/VAN-Tiny-original/resolve/main/van_tiny_754.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )
        strict = False
        del checkpoint["state_dict"]["head.weight"]
        del checkpoint["state_dict"]["head.bias"]
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return model



if __name__ =="__main__":

    device = torch.device("cuda")
    model = build_model(pretrained=True).to(device)


    img = torch.randn((4, 3, 512, 512)).to(device)
    outs = model(img)
    print(outs.shape)



