import torch.nn as nn


class SURELoss(nn.Module):
    def __init__(self, gama=0.2):
        super().__init__()
        self.gama = gama
        self.cls_loss = nn.SoftMarginLoss(reduction='mean')
        self.reg_loss = nn.MSELoss(reduction='mean')

    def forward(self, y1, y2, code_1, code_2, dt):
        if len(dt.shape) + 1 == len(y1.shape):
            dt = dt.view(-1, 1)
        # print(y1, code_1)
        cls = self.cls_loss(y1, code_1) + self.cls_loss(y2, code_2)
        reg = self.reg_loss(y2 - y1, dt.repeat(1, y1.size(1)))
        loss = cls + self.gama * reg
        return loss
