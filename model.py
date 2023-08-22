import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, kernel_type, kernel_x, kernel_y, in_dim, out_dim,
                 dropout=0.1, kernel_x2=None, kernel_y2=None):
        super().__init__()
        if kernel_type == 0:
            padding1 = kernel_x // 2
            padding2 = kernel_y // 2
            self.conv1 = nn.Conv2d(in_dim, out_dim//2, (kernel_x, 1), padding=(padding1, 1))
            self.conv2 = nn.Conv2d(out_dim//2, out_dim, (kernel_y, 1), padding=(padding2, 1))

        if kernel_type == 1:
            if kernel_x2 is None:
                kernel_x2 = kernel_x
            if kernel_y2 is None:
                kernel_y2 = kernel_y
            padding1 = (kernel_x // 2, kernel_y2 // 2)
            padding2 = (kernel_y // 2, kernel_x2 // 2)
            self.conv1 = nn.Conv2d(in_dim, out_dim//2, (kernel_x, kernel_y2), padding=padding1)
            self.conv2 = nn.Conv2d(out_dim//2, out_dim, (kernel_y, kernel_x2), padding=padding2)

        if kernel_type == 2:
            padding1 = kernel_x // 2
            padding2 = kernel_y // 2
            self.conv1 = nn.Conv2d(in_dim, out_dim//2, kernel_x, padding=padding1)
            self.conv1 = nn.Conv2d(out_dim//2, out_dim, kernel_y, padding=padding2)

        self.norm1 = nn.BatchNorm2d(out_dim//2)
        self.norm2 = nn.BatchNorm2d(out_dim)
        self.nonlinear = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.nonlinear(y)
        y = self.drop(y)
        y = y.transpose(3, 2)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.nonlinear(y)
        y = self.drop(y)
        y = y.transpose(3, 2)
        y = self.pool(y)
        return y


class Model(nn.Module):
    def __init__(self, kernel_type=0, kernel_x=5, kernel_y=5, num_layers=3, hidden_dim=64, channels=16,
                 input_size=51, seq_len=10, pred_len=10):
        """ Model initializer """
        super().__init__()

        self.encoder = nn.Sequential(nn.Conv1d(seq_len, hidden_dim, 1), nn.Linear(input_size, hidden_dim))

        layers = []
        for i in range(num_layers):
            layers.append(ConvBlock(kernel_type, kernel_x, kernel_y, in_dim=1 if i == 0 else channels * 2 ** (i - 1),
                                    out_dim=channels * 2 ** i))

        self.conv_blocks = nn.Sequential(*layers)

        self.decoder = nn.Sequential(nn.Conv2d(channels * 2 ** (num_layers - 1), 1, 1),
                                     nn.AdaptiveMaxPool2d(1),
                                     nn.Flatten(1, 2),
                                     nn.Conv1d(1, pred_len, 1),
                                     nn.Linear(1, input_size)
                                     )

        return

    def forward(self, x):
        y = self.encoder(x)
        y = y.unsqueeze(1)
        y = self.conv_blocks(y)
        out = self.decoder(y)
        return out
