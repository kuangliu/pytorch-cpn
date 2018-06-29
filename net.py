import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, out_planes=0, stride=1):
        super(Bottleneck, self).__init__()
        out_planes = self.expansion*planes if out_planes == 0 else out_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class CPNet(nn.Module):
    def __init__(self, num_blocks, num_keypoints=17):
        super(CPNet, self).__init__()
        self.in_planes = 64
        self.num_keypoints = num_keypoints

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer( 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.lateral1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral4 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.global1 = self._make_global(scale_factor=8)
        self.global2 = self._make_global(scale_factor=4)
        self.global3 = self._make_global(scale_factor=2)
        self.global4 = self._make_global(scale_factor=1)

        self.refine1 = self._make_refine(num_blocks=3, scale_factor=8)
        self.refine2 = self._make_refine(num_blocks=2, scale_factor=4)
        self.refine3 = self._make_refine(num_blocks=1, scale_factor=2)
        self.refine4 = nn.Sequential(
            Bottleneck(4*256, 128, 256),
            nn.Conv2d(256, num_keypoints, kernel_size=3, stride=1, padding=1),
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride=stride))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)

    def _make_global(self, scale_factor):
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(256, self.num_keypoints, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
        )

    def _make_refine(self, num_blocks, scale_factor):
        layers = []
        for i in range(num_blocks):
            layers.append(Bottleneck(256,128,256))
        layers.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False))
        return nn.Sequential(*layers)

    def _upsample_smooth_add(self, x, smooth, y):
        up = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        return smooth(up) + F.relu(y)

    def forward(self, x):
        # Top-down
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Bottom-up
        p5 = self.lateral1(c5)
        p4 = self._upsample_smooth_add(p5, self.smooth1, self.lateral2(c4))
        p3 = self._upsample_smooth_add(p4, self.smooth2, self.lateral3(c3))
        p2 = self._upsample_smooth_add(p3, self.smooth3, self.lateral4(c2))
        # GlobalNet
        g5 = self.global1(p5)
        g4 = self.global2(p4)
        g3 = self.global3(p3)
        g2 = self.global4(p2)
        # RefineNet
        r5 = self.refine1(p5)
        r4 = self.refine2(p4)
        r3 = self.refine3(p3)
        r2 = p2
        r = torch.cat([r5,r4,r3,r2], 1)
        r = self.refine4(r)
        return g5, g4, g3, g2, r


def CPNet50():
    return CPNet([3,4,6,3])

def CPNet101():
    return CPNet([3,4,23,3])

def CPNet152():
    return CPNet([3,8,36,3])


def test():
    net = CPNet50()
    ys = net(torch.randn(1,3,192,256))
    for y in ys:
        print(y.size())

test()
