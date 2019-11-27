import torch
import torch.nn as nn
import torch.nn.functional as F


class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, ]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),  # 卷积层中的前两个参数是图片channel的变化
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),     # 表示输入的是3channel，输出的是6channel
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
    #  打平的操作在后面的forward里面写，这里不写
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        # [b, 16, 5, 5]
        print('conv out:', out.shape)

        self.criteon = nn.CrossEntropyLoss()

    def forward(self, x): #
        batchsz = x.size(0)
        x = self.conv_unit(x)
        # [b,3,32,32] => [b,16,5,5]
        x = x.view(batchsz, 16*5*5)  # [b, 16*5*5]
        logits = self.fc_unit(x)

        # pred = F.softmax()
        # loss = self.criteon(logits, y)
        return logits


def main():

    net = Lenet5()

    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    # [b, 16, 5, 5]
    print('lenet out:', out.shape)


if __name__ == '__main__':
    main()