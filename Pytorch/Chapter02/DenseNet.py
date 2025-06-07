import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, input_num_planes, rate_inc):
        super(DenseBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(input_num_planes)
        self.conv_layer1 = nn.Conv2d(
            in_channels=input_num_planes,
            out_channels=4 * rate_inc,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(4 * rate_inc)
        self.conv_layer2 = nn.Conv2d(
            in_channels=4 * rate_inc,
            out_channels=rate_inc,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, inp):
        op = self.conv_layer1(F.relu(self.batch_norm1(inp)))
        op = self.conv_layer2(F.relu(self.batch_norm2(op)))
        op = torch.cat([op, inp], 1)
        return op


class TransBlock(nn.Module):
    def __init__(self, input_num_planes, output_num_planes):
        super(TransBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(input_num_planes)
        self.conv_layer = nn.Conv2d(
            in_channels=input_num_planes,
            out_channels=output_num_planes,
            kernel_size=1,
            bias=False,
        )

    def forward(self, inp):
        op = self.conv_layer(F.relu(self.batch_norm(inp)))
        op = F.avg_pool2d(op, 2)
        return op
