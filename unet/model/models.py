"""
Defining Models
Code refactored from: https://github.com/milesial/Pytorch-UNet/tree/master semantic
segmentation implementation
"""

from model.unet_modules import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout = dropout

        self.inc = (DoubleConv(n_channels, 32, dropout))     # input image size selecting 32 as smallest
        self.down1 = (Down(32, 64, dropout))                 # doubling feature channels
        self.down2 = (Down(64, 128, dropout))                # doubling feature channels
        self.down3 = (Down(128, 256, dropout))               # doubling feature channels
        self.down4 = (Down(256, 512 // 2, dropout))
        self.up1 = (Up(512, 256 // 2, dropout))              # upsampling, halving number of features
        self.up2 = (Up(256, 128 // 2, dropout))              # upsampling, halving number of features
        self.up3 = (Up(128, 64 // 2, dropout))               # upsampling, halving number of features
        self.up4 = (Up(64, 32, dropout))                     # supsampling, halving the number of features
        self.outc = (OutConv(32, n_classes))        # final output matches input size with number of classes specified (1 for regressiom)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        output = self.outc(x9)
        return output


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)