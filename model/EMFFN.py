import torch
import torch.nn as nn
import torch.nn.functional as F
from CDCN import CDCN
from PMN import PMN


class EMFFN(nn.Module):
    def __init__(self):
        super(EMFFN, self).__init__()
        self.CDCN_fc = CDCN()
        self.PWN_fc = PWN(5, 5, 5)
        self.FC = nn.Linear(14768, 9)

        # for m in self.modules():
        #     for name, parameter in m.named_parameters():
        #         if parameter.dim() > 1:
        #             # nn.init.normal_(parameter, mean=0, std=0.1)
        #             # nn.init.kaiming_uniform_(parameter, mode='fan_out', nonlinearity='relu')
        #             nn.init.kaiming_normal_(parameter, mode='fan_out', nonlinearity='relu')
        #         elif parameter.dim() == 1:
        #             if name.split('.')[-1] == 'weight':
        #                 # nn.init.normal_(parameter, mean=0, std=0.1)
        #                 nn.init.constant_(parameter, 0.1)
        #             elif name.split('.')[-1] == 'bias':
        #                 nn.init.constant_(parameter, 0)
        #     break

    def forward(self, x_in1, x_in2):
        out_CDCN = self.CDCN_fc(x_in1)[0]
        out_PWN = self.PWN_fc(x_in2)[0]
        out = torch.cat([
            out_CDCN, out_PWN
        ], dim=1)
        # fc_all=torch.Size([32, 81584])

        # 进行全连接
        out = self.FC(out)

        return out


def main():
    net = EMFFN()

    tmp1 = torch.randn(32, 103, 25, 25)
    tmp2 = torch.randn(32, 5, 25, 25)

    out = net(tmp1, tmp2)
    print(out.shape)


if __name__ == '__main__':
    main()
