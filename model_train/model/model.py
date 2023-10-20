import torch.nn as nn


class ExampleModel(nn.Module):
    def __init__(self, input_channel=1, class_num=10) -> None:
        super(ExampleModel, self).__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.module2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(7 * 7 * 64, class_num, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.module1(x)
        out = self.module2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out
