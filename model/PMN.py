import torch
import torch.nn as nn
import torch.nn.functional as F


# 图像在进来的数据就已经是patch了
# torch(32,1,27,27)
# 根据模块图进行切分为5个部分，1+3+1，1为3的组合大体，进行3次调用，最后通过1来进行特征融合


class Conv_module1(nn.Module):
    def __init__(self, in_channels):
        super(Conv_module1, self).__init__()
        self.conv_stage1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
        )
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        out1 = self.conv_stage1(x)
        out2 = self.conv1(x)

        return out1, out2


class Conv_module2(nn.Module):
    def __init__(self, in_channels):
        super(Conv_module2, self).__init__()
        self.conv_stage2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        out1 = self.conv_stage2(x)
        out2 = self.conv2(x)

        return out1, out2


class Conv_module3(nn.Module):
    def __init__(self, in_channels):
        super(Conv_module3, self).__init__()
        self.conv_stage3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        out = self.conv_stage3(x)

        return out


class Conv_stage(nn.Module):
    def __init__(self, in_channel_one, in_channel_two, in_channel_three):
        super(Conv_stage, self).__init__()
        self.Con_part1 = Conv_module1(in_channels=in_channel_one)
        self.Con_part2 = Conv_module2(in_channels=in_channel_two)
        self.Con_part3 = Conv_module3(in_channels=in_channel_three)
        self.Max_Pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x_in1, x_in2, x_in3):
        out1_model = self.Con_part1(x_in1)[0]
        out1_org = self.Con_part1(x_in1)[1]
        out2_model = self.Con_part2(x_in2)[0]
        out2_org = self.Con_part2(x_in2)[1]
        out3_model = self.Con_part3(x_in3)

        out2 = torch.cat([out2_model, out1_org], dim=1)
        out3 = torch.cat([out1_org, out2_org, out3_model], dim=1)

        out1 = self.Max_Pool2d(out1_model)
        out2 = self.Max_Pool2d(out2)
        out3 = self.Max_Pool2d(out3)

        return out1, out2, out3


class PMN(nn.Module):
    def __init__(self, in_channel_one, in_channel_two, in_channel_three):
        super(PMN, self).__init__()
        self.Conv_stage1 = Conv_stage(in_channel_one, in_channel_two, in_channel_three)
        self.Conv_stage2 = Conv_stage(16, 16 * 3, 16 * 7)
        self.Conv_stage3 = Conv_stage(16, 16 * 3, 16 * 7)
        self.Flatten = nn.Flatten()
        self.outlayer = nn.Linear(1584, 16)

    def forward(self, x):
        # 输出成为3种特征图
        out_stage1 = self.Conv_stage1(x, x, x)
        out_stage2 = self.Conv_stage2(out_stage1[0], out_stage1[1], out_stage1[2])
        out_stage3 = self.Conv_stage2(out_stage2[0], out_stage2[1], out_stage2[2])
        # torch.Size([32, 16, 3, 3]) torch.Size([32, 48, 3, 3]) torch.Size([32, 112, 3, 3])
        # return out_stage3[0], out_stage3[1], out_stage3[2]

        out_fc1 = self.Flatten(out_stage3[0])
        out_fc2 = self.Flatten(out_stage3[1])
        out_fc3 = self.Flatten(out_stage3[2])

        out_FcALL = torch.cat([out_fc1, out_fc2, out_fc3], dim=1)
        out_FcALL_sg = torch.sigmoid(out_FcALL)

        out = out_FcALL.mul(out_FcALL_sg)
        out_org = torch.relu(out)

        # #进行全连接
        out = self.outlayer(out_org)

        return out_org, out


def main():
    net = PMN(5, 5, 5)

    tmp1 = torch.randn(128, 5, 25, 25)

    out = net(tmp1)[1]
    print(out.shape)


# （2）测试代码Conv_stage
# net = Conv_stage(1, 1, 1)
#
# tmp1 = torch.randn(32, 1, 27, 27)
# tmp2 = torch.randn(32, 1, 27, 27)
# tmp3 = torch.randn(32, 1, 27, 27)
#
# out1 = net(tmp1, tmp2, tmp3)[0]
# out2 = net(tmp1, tmp2, tmp3)[1]
# out3 = net(tmp1, tmp2, tmp3)[2]
#
# print(out1.shape, out2.shape, out3.shape)


# （1）测试代码Conv_module1
#     net = Conv_module3(5)
#     tmp = torch.randn(32, 5, 27, 27)
#     out1 = net(tmp)[0]
#     out2 = net(tmp)[1]
#
#     print(out1.shape, out2.shape)

if __name__ == '__main__':
    main()
