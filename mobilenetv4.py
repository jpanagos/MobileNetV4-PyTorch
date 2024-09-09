import torch.nn as nn

class UniversalInvertedBottleneck(nn.Module):
    def __init__(self, input_size, expanded_dim, output_size, extra_dw_ks=None, int_dw_ks=None, stride=1, fused=False):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride

        int_downsample = int_dw_ks is not None
        first_stride = stride if not int_downsample else 1
        int_stride = stride if int_downsample else 1

        if extra_dw_ks is not None:
            padding = (extra_dw_ks - 1) // 2
            self.first_dw = nn.Conv2d(input_size, input_size, kernel_size=extra_dw_ks, stride=first_stride,
                                      padding=padding, groups=input_size, bias=False)
            self.first_bn = nn.BatchNorm2d(input_size)
        else:
            self.first_dw = nn.Identity()
            self.first_bn = nn.Identity()

        if fused:
            padding = (extra_dw_ks - 1) // 2
            self.expand = nn.Conv2d(input_size, expanded_dim, kernel_size=extra_dw_ks, stride=first_stride,
                                    padding=padding, bias=False)
            self.first_dw = nn.Identity()
            self.first_bn = nn.Identity()
        else:
            self.expand = nn.Conv2d(input_size, expanded_dim, kernel_size=1, stride=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_dim)
        self.expand_act = nn.ReLU()

        if int_dw_ks is not None:
            padding = (int_dw_ks - 1) // 2
            self.int_dw = nn.Conv2d(expanded_dim, expanded_dim, kernel_size=int_dw_ks, stride=int_stride,
                                    padding=padding, groups=expanded_dim, bias=False)
            self.int_bn = nn.BatchNorm2d(expanded_dim)
            self.int_act = nn.ReLU()
        else:
            self.int_dw = nn.Identity()
            self.int_bn = nn.Identity()
            self.int_act = nn.Identity()

        self.proj = nn.Conv2d(expanded_dim, output_size, kernel_size=1, stride=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(output_size)

        ## ending dw conv (conv-bn, stride=1)

    def forward(self, inputs):
        x = self.first_bn(self.first_dw(inputs))
        x = self.expand_act(self.expand_bn(self.expand(x)))
        x = self.int_act(self.int_bn(self.int_dw(x)))
        if self.input_size == self.output_size and self.stride == 1:
            return self.bn_proj(self.proj(x)) + inputs
        return self.bn_proj(self.proj(x))

class MobileNetV4(nn.Module):
    def __init__(self, in_channels, out_channels, block_spec, has_stem, has_classifier):
        super().__init__()
        if has_stem:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU()
            )
        else:
            self.stem = nn.Identity()
        self.blocks = nn.Sequential()
        in_channels = 32
        for block in block_spec:
            expanded_dim, output_size, extra_dw_ks, int_dw_ks, stride, fused = self._decode_block(block)
            _block = UniversalInvertedBottleneck(in_channels, expanded_dim, output_size, extra_dw_ks, int_dw_ks, stride, fused)
            self.blocks.append(_block)
            in_channels = output_size
        self.head = nn.Sequential(nn.Conv2d(in_channels, 960, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(960),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1)
        )
        if has_classifier:
            self.classifier = nn.Sequential(nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False),
                                            nn.ReLU(),
                                            nn.Conv2d(1280, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.classifier = nn.Identity()
        self._initialize_layers()

    def _decode_block(self, block):
        extra_dw_ks  = block[0]
        int_dw_ks    = block[1]
        expanded_dim = block[2]
        output_size  = block[3]
        stride       = block[-2]
        fused        = block[-1]
        return expanded_dim, output_size, extra_dw_ks, int_dw_ks, stride, fused

    def _initialize_layers(self): # following PyTorch MobileNetV2/3 initialization w/ "fan_out"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out") # official TF uses "fan_in"
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        return self.classifier(self.head(self.blocks(self.stem(inputs))))

def get_mobilenet_v4(inputs: int, outputs: int, variant: str, stem: bool, classifier: bool):
    if variant in ['s', 'S']:
        block_spec = [
            [3, 3, 32, 32, 2, True],
            [3, 3, 96, 64, 2, True],

            [   5, 5, 192, 96, 2, False],
            [None, 3, 192, 96, 1, False],
            [None, 3, 192, 96, 1, False],
            [None, 3, 192, 96, 1, False],
            [None, 3, 192, 96, 1, False],
            [3, None, 384, 96, 1, False],

            [   3, 3, 576, 128, 2, False],
            [   5, 5, 512, 128, 1, False],
            [None, 5, 512, 128, 1, False],
            [None, 5, 384, 128, 1, False],
            [None, 3, 512, 128, 1, False],
            [None, 3, 512, 128, 1, False],
        ]
    elif variant in ['m', 'M']:
        block_spec = [
            [3, 3, 128, 48, 2, True],

            [3, 5, 192, 80, 2, False],
            [3, 3, 160, 80, 1, False],

            [   3,    5, 480, 160, 2, False],
            [   3,    3, 640, 160, 1, False],
            [   3,    3, 640, 160, 1, False],
            [   3,    5, 640, 160, 1, False],
            [   3,    3, 640, 160, 1, False],
            [   3, None, 640, 160, 1, False],
            [None, None, 320, 160, 1, False],
            [   3, None, 640, 160, 1, False],

            [   5,    5,  960, 256, 2, False],
            [   5,    5, 1024, 256, 1, False],
            [   3,    5, 1024, 256, 1, False],
            [   3,    5, 1024, 256, 1, False],
            [None, None, 1024, 256, 1, False],
            [   3, None, 1024, 256, 1, False],
            [   3,    5,  512, 256, 1, False],
            [   5,    5, 1024, 256, 1, False],
            [None, None, 1024, 256, 1, False],
            [None, None, 1024, 256, 1, False],
            [   5, None,  512, 256, 1, False],
        ]
    elif variant in ['l', 'L']:
        block_spec = [
            [3, 3, 96, 48, 2, True],

            [3, 5, 192, 96, 2, False],
            [3, 3, 384, 96, 1, False],
            [3,    5, 384, 192, 2, False],
            [3,    3, 768, 192, 1, False],
            [3,    3, 768, 192, 1, False],
            [3,    3, 768, 192, 1, False],
            [3,    5, 768, 192, 1, False],
            [5,    3, 768, 192, 1, False],
            [5,    3, 768, 192, 1, False],
            [5,    3, 768, 192, 1, False],
            [5,    3, 768, 192, 1, False],
            [5,    3, 768, 192, 1, False],
            [3, None, 768, 192, 1, False],

            [5,    5,  768, 512, 2, False],
            [5,    5, 2048, 512, 1, False],
            [5,    5, 2048, 512, 1, False],
            [5,    5, 2048, 512, 1, False],
            [5, None, 2048, 512, 1, False],
            [5,    3, 2048, 512, 1, False],
            [5, None, 2048, 512, 1, False],
            [5, None, 2048, 512, 1, False],
            [5,    3, 2048, 512, 1, False],
            [5,    5, 2048, 512, 1, False],
            [5, None, 2048, 512, 1, False],
            [5, None, 2048, 512, 1, False],
            [5, None, 2048, 512, 1, False],
        ]
    else:
        raise ValueError('Unrecognized model architecture. Options: s/S, m/M, l/L.')
    return MobileNetV4(inputs, outputs, block_spec, stem, classifier)
