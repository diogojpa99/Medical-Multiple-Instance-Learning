"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, \
    get_act_layer, get_norm_layer, create_classifier
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, model_entrypoint, generate_default_cfgs, register_model_deprecations
 
__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
    ):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
    ):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        first_dilation=None,
        norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        first_dilation=None,
        norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]

def make_blocks(
        block_fn,
        channels,
        block_repeats,
        inplanes,
        reduce_first=1,
        output_stride=32,
        down_kernel_size=1,
        avg_down=False,
        drop_block_rate=0.,
        drop_path_rate=0.,
        **kwargs,
):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None,
                **block_kwargs,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info

class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            cardinality=1,
            base_width=64,
            stem_width=64,
            stem_type='',
            replace_stem_pool=False,
            block_reduce_first=1,
            down_kernel_size=1,
            avg_down=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            aa_layer=None,
            drop_rate=0.0,
            drop_path_rate=0.,
            drop_block_rate=0.,
            zero_init_last=True,
            block_args=None,
            desired_output_size = 14,
            feature_extractor = False
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        
        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True),
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        #self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)
        
        self.desired_output_size = desired_output_size
        self.feature_extractor = feature_extractor

    @staticmethod
    def from_pretrained(model_name: str, load_weights=True, **kwargs) -> 'ResNet':
        entry_fn = model_entrypoint(model_name, 'resnet')
        return entry_fn(pretrained=not load_weights, **kwargs)

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)      

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            if self.desired_output_size == 14: 
                return x
            #x = self.layer4(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        
        """ if not self.feature_extractor:
            x = self.forward_head(x) """
            
        return x


def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


def _tcfg(url='', **kwargs):
    return _cfg(url=url, **dict({'interpolation': 'bicubic'}, **kwargs))


def _ttcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'test_input_size': (3, 288, 288), 'test_crop_pct': 0.95,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models',
    }, **kwargs))


def _rcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'crop_pct': 0.95, 'test_input_size': (3, 288, 288), 'test_crop_pct': 1.0,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models', 'paper_ids': 'arXiv:2110.00476'
    }, **kwargs))


def _r3cfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'input_size': (3, 160, 160), 'pool_size': (5, 5),
        'crop_pct': 0.95, 'test_input_size': (3, 224, 224), 'test_crop_pct': 0.95,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models', 'paper_ids': 'arXiv:2110.00476',
    }, **kwargs))


def _gcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic',
        'origin_url': 'https://cv.gluon.ai/model_zoo/classification.html',
    }, **kwargs))


default_cfgs = generate_default_cfgs({
    # ResNet
    'resnet18.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet18-5c106cde.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet34.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet50.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet50.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet101.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet101.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet101-cd907fc2.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet152.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet152.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet152-f82ba261.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet50_2.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet50_2.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet101_2.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet101_2.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
})

@register_model
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    #model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    return _create_resnet('resnet18', pretrained, **dict(**kwargs))

@register_model
def resnet34(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3])
    return _create_resnet('resnet34', pretrained, **dict(model_args, **kwargs))

@register_model
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    #model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50', pretrained, **dict(**kwargs))

@register_model
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3])
    return _create_resnet('resnet101', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3])
    return _create_resnet('resnet152', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet200(pretrained=False, **kwargs):
    """Constructs a ResNet-200 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3])
    return _create_resnet('resnet200', pretrained, **dict(model_args, **kwargs))


register_model_deprecations(__name__, {
    'tv_resnet18': 'resnet18.tv_in1k',
    'tv_resnet34': 'resnet34.tv_in1k',
    'tv_resnet50': 'resnet50.tv_in1k',
    'tv_resnet101': 'resnet101.tv_in1k',
    'tv_resnet152': 'resnet152.tv_in1k',
})