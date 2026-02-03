import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FEA(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        nn.init.zeros_(self.conv.weight)  #

    def forward(self, x):
        # x: [B, C, H, W]
        attn = torch.sigmoid(self.conv(x))  # [B, 1, H, W]
        return x + x * attn
class DirectionSelector(nn.Module):
    """
    """
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_attn = FEE(in_channels)
        self.pool_fc_h = nn.Linear(in_channels, 1)
        self.pool_fc_v = nn.Linear(in_channels, 1)

    def _pool_h(self, x):
        # x: [B, C, H, W]
        x_attn = self.spatial_attn(x)
        x_h = x_attn.mean(dim=2)       # [B, C, W]
        x_h = x_h.permute(0, 2, 1)     # [B, W, C]
        w_h = torch.sigmoid(self.pool_fc_h(x_h))  # [B, W, 1]
        pooled_h = (x_h * w_h).sum(dim=1)        # [B, C]
        return pooled_h

    def _pool_v(self, x):
        x_attn = self.spatial_attn(x)
        x_v = x_attn.mean(dim=3)       # [B, C, H]
        x_v = x_v.permute(0, 2, 1)     # [B, H, C]
        w_v = torch.sigmoid(self.pool_fc_v(x_v))
        pooled_v = (x_v * w_v).sum(dim=1)
        return pooled_v

    def forward(self, x):
        f_h = self._pool_h(x)
        f_v = self._pool_v(x)
        return 0.5 * (f_h + f_v)  # [B, C]



class DirectionalAttentionPooling(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.attn_fc = nn.Linear(in_channels, 1)

    def _pool_h(self, x):
        # x: [B, C, H, W] → [B, C, W]
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)  # [B, W, C]
        w = torch.softmax(self.attn_fc(x), dim=1)
        return (x * w).sum(dim=1)

    def _pool_v(self, x):
        # x: [B, C, H, W] → [B, C, H]
        x = x.mean(dim=3)
        x = x.permute(0, 2, 1)  # [B, H, C]
        w = torch.softmax(self.attn_fc(x), dim=1)
        return (x * w).sum(dim=1)

    def forward(self, x):
        f_h = self._pool_h(x)
        f_v = self._pool_v(x)
        return 0.5 * (f_h + f_v)
class LSRFE(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(LSRFE, self).__init__()
        # 1x1 卷积
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4 * growth_rate)
        self.relu1 = nn.ReLU(inplace=True)

        # 3x3 卷积
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):

        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        return torch.cat([x, x2], 1)  #
class LSRFEModule(nn.Module):

    def __init__(self, num_layers, in_channels, growth_rate):
        super(LSRFEModule, self).__init__()
        self.layers = nn.ModuleList([LSRFE(in_channels + i * growth_rate, growth_rate)
                                     for i in range(num_layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x


class NSNDP(nn.Module):
     def __init__(self, num_classes=2):
        super(NSNDP, self).__init__()


        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = LSRFEModule(6, 64, 32)
        self.trans1 = TransitionLayer(64 + 6 * 32, 128)

        self.block2 = LSRFEModule(12, 128, 32)
        self.trans2 = TransitionLayer(128 + 12 * 32, 256)


        self.block3 = LSRFEModule(24, 256, 32)
        self.trans3 = TransitionLayer(256 + 24 * 32, 512)


        self.block4 = LSRFEModule(16, 512, 32)

        self.spine_attn = FEA(512 + 16 * 32)
        self.dir_pool = DirectionSelector(512 + 16 * 32)


        self.avg_pool = nn.AdaptiveAvgPool2d(1)


        self.fc = nn.Linear(512 + 16 * 32, num_classes)

    def forward(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.block1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.trans2(x)

        x = self.block3(x)
        x = self.trans3(x)

        x = self.block4(x)
        x = self.spine_attn(x)
        x = self.dir_pool(x)

        x = self.fc(x)

        return x


def NSNDP_system(num_classes=2):
    return NSNDP(num_classes=num_classes)
