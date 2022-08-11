import torch
import torch.nn as nn
import torch.nn.functional as F


# (1)CDCN先进行将原始数据打平
# (2)将数据从Original Data 连成 Spectral Vector
# (3) torch(103, 610, 340)

# 空洞卷积网络
# pytorch中如果使用3*3的空洞卷积，且步长为1，此时设置padding=dilation，那么输出特征图与输入特征图大小相同。
########################################################################################################################
class Dilated_conv_block_beggining(nn.Module):
    def __init__(self, Rate):
        super(Dilated_conv_block_beggining, self).__init__()
        # 定义单层的空洞卷积形式
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=6, padding='same', dilation=Rate),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6, padding='same', dilation=Rate),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class Dilated_resnet_beggining(nn.Module):
    def __init__(self, Rate):
        super(Dilated_resnet_beggining, self).__init__()

        self.Dnet = Dilated_conv_block_beggining(Rate)
        self.extra = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=1, stride=1)
        )

        # 保证输出的通道数一致
        # 在这里通道数固定，如果不固定则需要相应的判断

    def forward(self, x):
        out1 = self.Dnet(x)
        out2 = self.extra(x)
        out = out1 + out2
        out = F.relu(out)

        return out


########################################################################################################################
class Dilated_conv_block(nn.Module):
    def __init__(self, Rate):
        super(Dilated_conv_block, self).__init__()
        # 定义单层的空洞卷积形式
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6, padding='same', dilation=Rate),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6, padding='same', dilation=Rate),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class Dilated_resnet(nn.Module):
    def __init__(self, Rate):
        super(Dilated_resnet, self).__init__()

        self.Dnet = Dilated_conv_block(Rate)
        self.extra = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        )

        # 保证输出的通道数一致
        # 在这里通道数固定，如果不固定则需要相应的判断

    def forward(self, x):
        out1 = self.Dnet(x)
        out2 = self.extra(x)
        out = out1 + out2
        out = F.relu(out)

        return out


########################################################################################################################
# 通过注意力机制让
class Senet_attention(nn.Module):
    def __init__(self):
        super(Senet_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 将w,h打平成为1X1
        self.fc = nn.Sequential(
            nn.Linear(128, 5, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(5, 128, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


########################################################################################################################
class CDCN(nn.Module):
    def __init__(self):
        super(CDCN, self).__init__()
        # 先对图像进行1X1的卷积
        #
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1)
        self.D1 = Dilated_conv_block_beggining(1)
        self.D1_block = Dilated_resnet_beggining(1)
        self.D2 = Dilated_conv_block(2)
        self.D2_block = Dilated_resnet(2)
        self.D3 = Dilated_conv_block(4)
        self.D3_block = Dilated_resnet(4)
        self.D4 = Dilated_conv_block(8)
        self.D4_block = Dilated_resnet(8)

        self.Flatten = nn.Flatten()
        self.attention = Senet_attention()
        self.outlayer = nn.Linear(13184, 9)

    def forward(self, x):
        b, c, w, h = x.shape
        input = x[:, :, w // 2, h // 2]
        input = input.unsqueeze(dim=1)
        out = self.conv1(input)

        out1 = self.D1_block(out)
        out2 = self.D2_block(out1)
        out3 = self.D3_block(out2)
        out4 = self.D4_block(out3)

        out_cat1 = self.D1(out)
        out_cat2 = self.D2(out1)
        out_cat3 = self.D3(out2)
        # print("finish!", out_cat1.shape, out_cat2.shape, out_cat3.shape)
        out = torch.cat([
            out_cat1,
            out_cat2,
            out_cat3,
            out4,
        ], dim=1)

        out = self.attention(out)
        out_org = self.Flatten(out)
        out = self.outlayer(out_org)

        return out_org, out


def main():
    net = CDCN()
    tmp = torch.randn(32, 103, 25, 25)
    out = net(tmp)[0]
    print(out.shape)

    # net = Dilated_conv_block(8)
    # tmp = torch.randn(32, 32, 1)
    # out = net(tmp)
    # print(out.shape)


if __name__ == '__main__':
    main()
