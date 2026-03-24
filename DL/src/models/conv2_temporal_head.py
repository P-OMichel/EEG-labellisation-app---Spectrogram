'''
Encoder: block element is: {conv2D + conv2D + batchnorm + relu} and is repeated
Head: temporal head
'''

import torch
import torch.nn as nn

from .registry import register_model


#------------------------ Block elements ---------------------
class Encoder(nn.Module):
    def __init__(self, in_ch: int , enc_ch: int ):
        super().__init__()

        self.net = nn.Sequential(
            # ---- Block 1 (2 convs, then pool) ----
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(kernel_size=2, stride=2),  # H,W -> H/2,W/2 # NOTE downsample also in time which makes it harder for loss since mask is not reduced in length
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # only H -> H/2, keep W

            # ---- Block 2 (2 convs, then pool) ----
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        
            #nn.MaxPool2d(kernel_size=2, stride=2),  # H,W -> H/4,W/4
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # only H -> H/2, keep W

            # ---- Final projection to enc_ch (no pooling, same as your original) ----
            nn.Conv2d(64, enc_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TemporalHeadTCN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, hidden: int , layers: int , k: int = 3, dropout: float = 0.1):
        super().__init__()
        blocks = []
        c = in_ch
        for i in range(layers):
            d = 2 ** i
            blocks.append(nn.Conv1d(c, hidden, kernel_size=k, padding=d*(k-1)//2, dilation=d))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(dropout))
            c = hidden
        self.tcn = nn.Sequential(*blocks)
        self.out = nn.Conv1d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.tcn(x))
    

#------------------------ Full Architecture ---------------------
@register_model("conv2_temporal")
class conv2Temporal(nn.Module):
    def __init__(self, num_classes: int = 10, enc_ch: int = 128, tcn_hidden: int = 128, tcn_layers: int = 3):
        super().__init__()
        self.encoder = Encoder(in_ch=1, enc_ch=enc_ch)
        self.head = TemporalHeadTCN(in_ch=enc_ch, num_classes=num_classes, hidden=tcn_hidden, layers=tcn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = z.mean(dim=2)
        return self.head(z)
