"""
Defining Models
Code refactored from: https://github.com/milesial/Pytorch-UNet/tree/master semantic
segmentation implementation
"""

from model.unet_modules import *
import torch

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


class UNetEmbedding(nn.Module):
    def __init__(self, n_channels, n_classes, dropout):
        super(UNetEmbedding, self).__init__()
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
        return x9


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


class DistrictClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, 1)

    def masked_avg_pool(self, emb, mask):
        """
        emb: (C, H, W)
        mask: (H, W)
        """
        mask_torch = torch.from_numpy(mask)  # shape (H, W)
        masked = emb * mask_torch
        total = mask.sum()  # Number of pixels in the region
        if total == 0:
            return torch.zeros(emb.shape[0], device=emb.device)
        return masked.sum(dim=(1, 2)) / total

    def forward(self, embeddings, district_masks):
            """
            embeddings: B,C,H,W
            district_masks: H,W
            labels: B,C,H,W
            """
            district_embedding_list = []
            for i in range(embeddings.shape[0]):
                pooled_emb = []
                for district in district_masks.keys():
                    pooled_emb.append(self.masked_avg_pool(embeddings[i, :, :, :], district_masks[district]))
                pooled_emb = torch.stack(pooled_emb)
                district_embedding_list.append(pooled_emb)

            torch_embeddings = torch.stack(district_embedding_list)
            logits = self.classifier(torch_embeddings)

            return logits


class UNetDistrict(nn.Module):
    def __init__(self, n_channels, n_classes, dropout, embedding_dim, district_masks):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.unet = UNetEmbedding(n_channels, n_classes, dropout)
        self.district_classifier = DistrictClassifier(embedding_dim)
        self.district_masks = district_masks

    def forward(self, x):
        embeddings = self.unet(x)  # (B, C, H, W)
        district_logits = self.district_classifier(embeddings, self.district_masks)
        return embeddings, district_logits


class UNetMini(nn.Module):
    def __init__(self, n_channels, n_classes, dropout):
        super(UNetMini, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout = dropout

        self.inc = (DoubleConv(n_channels, 32, dropout))     # input image size selecting 32 as smallest
        self.down1 = (Down(32, 64, dropout))                 # doubling feature channels
        self.down2 = (Down(64, 128, dropout))                # doubling feature channels
        self.down3 = (Down(128, 256 // 2, dropout))
        self.up1 = (Up(256, 128 // 2, dropout))              # upsampling, halving number of features
        self.up2 = (Up(128, 64 // 2, dropout))              # upsampling, halving number of features
        self.up3 = (Up(64, 32, dropout))                    # upsampling, halving the number of features
        self.outc = (OutConv(32, n_classes))        # final output matches input size with number of classes specified (1 for regressiom)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        output = self.outc(x7)
        return output


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)