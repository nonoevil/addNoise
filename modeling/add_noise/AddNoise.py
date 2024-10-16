from xml.sax import make_parser
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random
from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert
import numpy as np

class AddNoise(nn.Module):
    def __init__(
            self,
            prompt_channel = 1,
            image_channel = 512,
            appearance_guidance_dims = [512,256,128]
    ):
        # prompt_channel: Number of prompts for ensembling text features. Default: 1
        super().__init__()

        self.mean = 0.0  # 高斯噪声的均值
        self.stddev = 0.00002  # 高斯噪声的标准差
        # self.cfg = cfg
        # self.addapter = nn.Sequential(
        #     nn.Conv2d(prompt_channel, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        # self.mse_loss = nn.MSELoss()
        # self.GetImageCls = nn.Conv2d(image_channel, image_channel, kernel_size=3, stride=1, padding=1)
        # self.GetImageNoCls = nn.Conv2d(image_channel, image_channel, kernel_size=3, stride=1, padding=1)
        self.Conv1NoiseToImage = nn.Sequential(
            nn.Conv2d(image_channel, 768, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.Conv2NoiseToImage = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # self.conv = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(image_channel, appearance_guidance_dims[0], kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #     ) ,
        #     nn.Sequential(
        #         nn.ConvTranspose2d(image_channel, appearance_guidance_dims[1], kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #     ),
        #     nn.Sequential(
        #         nn.ConvTranspose2d(image_channel, appearance_guidance_dims[1], kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(appearance_guidance_dims[1], appearance_guidance_dims[2], kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #     ),
        # ])

        # self.upsample1 = nn.Upsample(size=(48, 48), mode='bilinear', align_corners=False)
        # self.upsample2 = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        # self.conv4 = nn.Conv2d(image_channel  *2, image_channel, kernel_size=3, stride=1, padding=1)

        # self.sumImage = torch.zeros((4,512,24,24), dtype=torch.float32, device="cuda:0")
        # self.sumText = torch.zeros((4,171,1,512), dtype=torch.float32, device="cuda:0")
        # self.maxnI = -100
        # self.maxnT = -100
        # self.minnI = 100
        # self.minnT = 100
        # self.times = 0

    def nanId(self, mask,  targets):
        index = [
            torch.unique(targets[i]).long() if torch.unique(targets[i]).long()[-1].item() != 255
            else torch.unique(targets[i])[:-1].long()
            for i in range(targets.shape[0])
        ]

        mask_class = mask.sum(dim=(2, 3, 4))
        mask_class = (mask_class > 0.1).sum(dim=1)

        nan_id = []
        Nonan_id = []
        for i in range(targets.shape[0]):
            if (mask_class[i].item() != len(index[i])) or (len(index[i]) == 0):
                nan_id.append((i))
            else:
                Nonan_id.append(i)

        while len(nan_id) < targets.shape[0] // 2:
            selected_number = random.choice(Nonan_id)
            nan_id.append(selected_number)
            Nonan_id.remove(selected_number)
        # print(f"nan_id{nan_id}")
        # print(f"Nonan_id{Nonan_id}")
        del index ,mask_class
        return nan_id,Nonan_id

    def NoiseToImage(self, image_shape, noise, targets, device):
            """
                Args:
                    image (Tensor): Input image of shape (B, C, H, W).
                    noise (Tensor): Noise image of shape (B, T, P, C). (4,171, 1, 512)
                Returns:
                    noise (B,C,H,W)
                    mask (B,T,C,H,W)
                    Nonanid  list (B,)
            """
            shape = (0,1,24,24)
            targets = targets.float()
            mask = F.interpolate(targets.unsqueeze(1), size=(24, 24), mode='nearest')
            #
            mask = mask.long()
            # (4, 24, 24)


            # (4, 1, 24, 24)
            new_mask = torch.zeros((targets.shape[0], 256, 24, 24),  device=device)
            new_mask.scatter_(1, mask, 1)
            # (4, 256, 24, 24)
            new_mask = new_mask[:, :171, :, :]

            new_mask = new_mask.unsqueeze(2).repeat(1, 1, image_shape[1], 1, 1)
            # (4, 171, 768, 24, 24)

            # (4,171, 1,512) -> (4,577,768)

            noise = noise.permute(0, 1, 3, 2).unsqueeze(4)
            # (4, 171, 1, 512) -> (4, 171, 512, 1, 1)
            noise = noise.view(-1, 512, 1, 1)  # 变为 (4 * 171, 768, 1, 1)
            noise = F.interpolate(noise, size=(12, 12), mode='bilinear',align_corners=False)
            # (4 171, 512, 1, 1) -> (4 171, 512, 12, 12)
            noise = self.Conv1NoiseToImage(noise)
            # (4171, 512, 512, 12) -> (4 171, 768, 12, 12)
            noise = F.interpolate(noise, size=(24, 24), mode='bilinear', align_corners=False)
            # (4 171, 768, 12, 12) -> (4 171, 768, 24, 24)
            noise = self.Conv2NoiseToImage(noise)
            # (4,171, 768, 24, 24) -> (4 171, 768, 24, 24)

            noise = noise.view(targets.shape[0], 171, image_shape[1], 24, 24)
            noise = torch.mul(noise, new_mask)
            noise = noise.sum(dim=1)
            # (4, 768, 24, 24)
            noise = rearrange(noise, "B C H W -> B C (H W)")
            noise = F.interpolate(noise.unsqueeze(2), size=(1,577), mode='bilinear',).squeeze(2)
            # (4, 768 ,577)
            nan_id, Nonan_id = self.nanId(new_mask, targets)
            zero_tensor = torch.zeros((image_shape[1], image_shape[2]), device=device)

            for id in nan_id:
                noise[id] = torch.mul(zero_tensor, noise[id])
            return noise.permute(2,0,1)

    def caculateNan(self,index, mask):
        for i in range(len(index)):
            for j in range(len(index[i])):
                idx = index[i][j]
                self.nanClass[0][idx] += 1
                if mask[i][idx] == 0:
                    self.nanClass[1][idx] += 1


    def getnoise(self, shape, device):
        noise = torch.normal(self.mean, self.stddev, shape, device=device,dtype=torch.float32)
        return  noise

    def forward(self,  image_shape, text_shape, targets, device):
        """
        h * w
        Args:
            Arguments:
                img_feats: (B, C, H*W)
                                768 577
                text_feats: (B, T, P, C)
                               171 1  512


        Returns:
        """

        # Ori_corr = self.correlation(image_features, text_features)
        #

        text_noise = self.getnoise(text_shape, device) # B T P C

        image_noise = self.NoiseToImage(image_shape,text_noise, targets, device)



        # Now_corr = self.correlation(image_features,text_features)

        # loss4 = self.mse_loss(Ori_corr, Now_corr)
        # print(f"loss4: {loss4}")
        # print(f"text_noise: {text_noise}")
        # print(f"image_features: {image_features}")
        # print(f"text_features: {text_features}")


        return  image_noise,text_noise
















