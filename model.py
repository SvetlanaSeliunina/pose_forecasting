import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, seq_len, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(seq_len, seq_len // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(seq_len // r, seq_len, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, kernel_type, kernel_x, kernel_y, hidden_dim, seq_len, time_hid, joints_hid,
                 dropout=0.1, kernel_x2=None, kernel_y2=None):
        super().__init__()
        self.kernel_type = kernel_type
        if kernel_type == 0:
            padding1 = kernel_x // 2
            padding2 = kernel_y // 2
            conv1_in = nn.Conv1d(seq_len, time_hid, kernel_x, padding=padding1)
            conv1_out = nn.Conv1d(time_hid, seq_len, kernel_x, padding=padding1)
            conv2_in = nn.Conv1d(hidden_dim, joints_hid, kernel_y, padding=padding2)
            conv2_out = nn.Conv1d(joints_hid, hidden_dim, kernel_y, padding=padding2)

        if kernel_type == 1:
            if kernel_x2 is None:
                kernel_x2 = kernel_x
            if kernel_y2 is None:
                kernel_y2 = kernel_y
            padding1 = (kernel_x // 2, kernel_y2 // 2)
            padding2 = (kernel_y // 2, kernel_x2 // 2)
            conv1_in = nn.Conv2d(1, time_hid, (kernel_x, kernel_y2), padding=padding1)
            conv1_out = nn.Conv2d(time_hid, 1, (kernel_x, kernel_y2), padding=padding1)
            conv2_in = nn.Conv2d(1, joints_hid, (kernel_y, kernel_x2), padding=padding2)
            conv2_out = nn.Conv2d(joints_hid, 1, (kernel_y, kernel_x2), padding=padding2)

        if kernel_type == 2:
            kernel = min(kernel_x, kernel_y)
            padding = kernel // 2
            conv1_in = nn.Conv2d(1, time_hid, kernel, padding=padding)
            conv1_out = nn.Conv2d(time_hid, 1, kernel, padding=padding)
            conv2_in = nn.Conv2d(1, joints_hid, kernel, padding=padding)
            conv2_out = nn.Conv2d(joints_hid, 1, kernel, padding=padding)

        self.norm1 = nn.LayerNorm(hidden_dim)
        nonlinear1 = nn.ReLU()
        nonlinear2 = nn.ReLU()
        drop1 = nn.Dropout(dropout)
        drop2 = nn.Dropout(dropout)
        self.SpatioBlock = nn.Sequential(conv1_in,
                                         nonlinear1,
                                         drop1,
                                         conv1_out)
        self.se = SELayer(seq_len)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.TemporalBlock = nn.Sequential(conv2_in,
                                           nonlinear2,
                                           drop2,
                                           conv2_out)

    def forward(self, x):
        y = self.norm1(x)
        if self.kernel_type != 0:
            y = y.unsqueeze(1)
        y = self.SpatioBlock(y)
        if self.kernel_type != 0:
            y = y.squeeze(1)
        y = self.se(y)
        x = x + y
        y = self.norm2(x)
        y = y.transpose(1, 2)
        if self.kernel_type != 0:
            y = y.unsqueeze(1)
        y = self.TemporalBlock(y)
        if self.kernel_type != 0:
            y = y.squeeze(1)
        y = y.transpose(1, 2)
        y = self.se(y)
        x = x + y
        return x


class Model(nn.Module):
    def __init__(self, kernel_type=0, kernel_x=7, kernel_y=3, num_layers=4, hidden_dim=64,
                 input_size=51, seq_len=10, pred_len=10, time_hid=32, joints_hid=128):
        """ Model initializer """
        super().__init__()

        self.encoder = nn.Linear(input_size, hidden_dim)
        layers = []
        for i in range(num_layers):
            layers.append(ConvBlock(kernel_type, kernel_x, kernel_y, hidden_dim,
                                    seq_len, time_hid, joints_hid))

        self.conv_blocks = nn.Sequential(*layers)

        self.decoder = nn.Sequential(nn.Conv1d(seq_len, pred_len, 1),
                                     nn.Linear(hidden_dim, input_size))

        return

    def forward(self, x):
        y = self.encoder(x)
        y = self.conv_blocks(y)
        out = self.decoder(y)
        return out
