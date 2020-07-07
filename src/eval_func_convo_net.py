import torch
import torch.nn as nn
import torch.nn.functional as F

import chess

class EvalFuncConvoNet(torch.nn.Module):
    def __init__(self):
        super(EvalFuncConvoNet, self).__init__()
        self.conv1 = nn.Conv2d(13, 16, 3, stride=1, padding=1)  # 16 kernels of shape (3, 3, 1); output H,W are (8 - 3 + 2 * 1) / 1 + 1 = 8; (16, 8, 8)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d(0.1)
        self.fc = nn.Linear(64 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # shape (n, 8, 8, 13)
        # NHWC to NCHW
        x = x.permute(0, 3, 1, 2)
        # shape (n, 13, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        # shape (n, 16, 8, 8)
        x = F.relu(self.bn2(self.conv2(x)))
        # shape (n, 32, 8, 8)
        x = F.relu(self.bn3(self.conv3(x)))
        # shape (n, 64, 8, 8)
        # x = F.max_pool2d(x, 2)  # kernel_size=2, stride_size=kernel_size
        # # shape (n, 16, 4, 4)
        x = torch.flatten(x, start_dim=1)
        # shape (n, 16 * 8 * 8)
        x = self.sigmoid(self.fc(x)).squeeze(-1)
        # shape (n,)
        return x

    def loss(self, prediction, label):
        return F.mse_loss(prediction, label)

shift = 4 * torch.arange(8)
masks = torch.tensor([0xf], dtype=torch.int64) << shift

def board_tensor(board:chess.Board, one_hot=False, unsqueeze=False, half=False):
    '''
    Returns an (8,8) shape torch.tensor form of the board's ranks array
    '''
    t = ((torch.tensor(board.board().contents.ranks, dtype=torch.int64).reshape(8, 1) & masks) >> shift)
    if (one_hot):
        t = F.one_hot(t, num_classes=13)  # 13 = 6 white + 6 black + 1 empty
    if (unsqueeze):
        t = t.unsqueeze(0)
    if (half):
        t = t.half()
    else:
        t = t.float()
    return t
