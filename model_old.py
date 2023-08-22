import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, kernel_type, kernel_x, kernel_y, hidden_dim, seq_len, batch_size,
                 dropout=0.1, kernel_x2=None, kernel_y2=None):
        super().__init__()
        if kernel_type == 0:
            padding1 = kernel_x // 2
            padding2 = kernel_y // 2
            self.conv1 = nn.Conv1d(seq_len, seq_len, kernel_x, padding=padding1)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_y, padding=padding2)
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(hidden_dim)

        if kernel_type == 1:
            if kernel_x2 is None:
                kernel_x2 = kernel_x
            if kernel_y2 is None:
                kernel_y2 = kernel_y
            padding1 = (kernel_x // 2, kernel_y2 // 2)
            padding2 = (kernel_y // 2, kernel_x2 // 2)
            self.conv1 = nn.Conv2d(batch_size, batch_size, (kernel_x, kernel_y2), padding=padding1)
            self.conv2 = nn.Conv2d(batch_size, batch_size, (kernel_y, kernel_x2), padding=padding2)
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(hidden_dim)

        if kernel_type == 2:
            kernel = min(kernel_x, kernel_y)
            padding = kernel // 2
            self.conv1 = nn.Conv2d(batch_size, batch_size, kernel, padding=padding)
            self.conv2 = nn.Conv2d(batch_size, batch_size, kernel, padding=padding)
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(hidden_dim)

        self.nonlinear = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.nonlinear(y)
        y = self.drop(y)
        y = y.transpose(1, 2)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.nonlinear(y)
        y = self.drop(y)
        y = y.transpose(1, 2)
        return y


class Model(nn.Module):
    def __init__(self, kernel_type=0, kernel_x=7, kernel_y=3, num_layers=4, hidden_dim=64,
                 input_size=51, seq_len=10, pred_len=10, batch_size=64):
        """ Model initializer """
        super().__init__()

        # self.encoder = nn.Conv2d(1, hidden_dim, (1, input_size))
        self.encoder = nn.Linear(input_size, hidden_dim)
        layers = []
        for i in range(num_layers):
            layers.append(ConvBlock(kernel_type, kernel_x, kernel_y, hidden_dim, seq_len, batch_size))

        self.conv_blocks = nn.Sequential(*layers)

        self.decoder = nn.Sequential(nn.Conv1d(seq_len, pred_len, 1),
                                     nn.Linear(hidden_dim, input_size))

        return

    def forward(self, x):
        y = self.encoder(x)
        y = self.conv_blocks(y)
        out = self.decoder(y)
        return out
