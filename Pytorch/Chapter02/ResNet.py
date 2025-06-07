import torch.nn as nn


class BasicBlock(nn.Module):
    multiplier = 1

    def __init__(self, input_num_planes, num_planes, strd=1):
        super(BasicBlock, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=input_num_planes,
            out_channels=num_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(num_planes)
        self.conv_layer2 = nn.Conv2d(
            in_channels=num_planes,
            out_channels=num_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(num_planes)
        self.res_connnection = nn.Sequential()
        if strd > 1 or input_num_planes != self.multiplier * num_planes:
            self.res_connnection = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_num_planes,
                    out_channels=self.multiplier * num_planes,
                    kernel_size=1,
                    stride=strd,
                    bias=False,
                ),
                nn.BatchNorm2d(self.multiplier * num_planes),
            )

    def forward(self, inp):
        op = F.relu(self.batch_norm1(self.conv_layer1(inp)))
        op = self.batch_norm2(self.conv_layer2(op))
        op += self.res_connnection(inp)
        op = F.relu(op)
        return op
