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
        self.stddev = 0.0002  # 高斯噪声的标准差
        # self.cfg = cfg
        # self.addapter = nn.Sequential(
        #     nn.Conv2d(prompt_channel, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        self.mse_loss = nn.MSELoss()
        # self.GetImageCls = nn.Conv2d(image_channel, image_channel, kernel_size=3, stride=1, padding=1)
        # self.GetImageNoCls = nn.Conv2d(image_channel, image_channel, kernel_size=3, stride=1, padding=1)
        self.Conv1NoiseToImage = nn.Sequential(
            nn.Conv2d(image_channel, image_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.Conv2NoiseToImage = nn.Sequential(
            nn.Conv2d(image_channel, image_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(image_channel, d, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d in appearance_guidance_dims
        ])


        # self.conv4 = nn.Conv2d(image_channel*2, image_channel, kernel_size=3, stride=1, padding=1)

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
        return nan_id,Nonan_id

    def NoiseToImage(self, image, noise, targets):
            """
                Args:
                    image (Tensor): Input image of shape (B, C, H, W).
                    noise (Tensor): Noise image of shape (B, T, P, C). (4,171, 1, 512)
                Returns:
                    noise (B,C,H,W)
                    mask (B,T,C,H,W)
                    Nonanid  list (B,)
            """
            targets = targets.float()
            mask = F.interpolate(targets.unsqueeze(1), size=(image.shape[2], image.shape[3]), mode='nearest')
            #
            mask = mask.long()
            # (4, 24, 24)


            # (4, 1, 24, 24)
            new_mask = torch.zeros((targets.shape[0], 256, image.shape[2], image.shape[3]),  device=image.device)
            new_mask.scatter_(1, mask, 1)
            # (4, 256, 24, 24)
            new_mask = new_mask[:, :171, :, :]

            new_mask = new_mask.unsqueeze(2).repeat(1, 1, image.shape[1], 1, 1)
            # (4, 171, 512, 24, 24)

            noise = noise.permute(0, 1, 3, 2).unsqueeze(4)
            # (4, 171, 1, 512) -> (4, 171, 512, 1, 1)
            noise = noise.view(-1, 512, 1, 1)  # 变为 (4 * 171, 512, 1, 1)
            # print(f"nosise3 {noise[0][0][0][0]}")
            noise = F.interpolate(noise, size=(image.shape[2] // 2, image.shape[3] // 2), mode='bilinear',
                                  align_corners=False)
            # print(f"nosise4 {noise}")
            # (4 171, 512, 1, 1) -> (4 171, 512, 12, 12)
            noise = self.Conv1NoiseToImage(noise)
            # print("Convolution weights (kernels):\n", self.Conv1NoiseToImage[0].weight.data)
            # print(f"Gradient for {self.Conv1NoiseToImage}:\n{self.Conv1NoiseToImage[0].weight.grad}")
            # print(f"nosise5 {noise}")
            # (4171, 512, 12, 12) -> (4 171, 512, 12, 12)
            noise = F.interpolate(noise, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
            # print(f"nosise6 {noise[0][0][0][0]}")
            # (4 171, 512, 12, 12) -> (4 171, 512, 24, 24)
            noise = self.Conv2NoiseToImage(noise)
            # (4,171, 512, 24, 24) -> (4 171, 512, 24, 24)
            # print(f"noise2 :{noise[0][0][0][0]}")
            noise = noise.view(targets.shape[0], 171, 512, 24, 24)
            noise = torch.mul(noise, new_mask)
            noise = noise.sum(dim=1)
            # (4, 512, 24, 24)

            nan_id, Nonan_id = self.nanId(new_mask, targets)
            for id in nan_id:
                zero_tensor = torch.zeros((512, 24, 24), device=image.device)
                noise[id] = torch.mul(zero_tensor, noise[id])

            return noise, new_mask,Nonan_id

    def caculateNan(self,index, mask):
        for i in range(len(index)):
            for j in range(len(index[i])):
                idx = index[i][j]
                self.nanClass[0][idx] += 1
                if mask[i][idx] == 0:
                    self.nanClass[1][idx] += 1


    def caculateLoss1(self, ImageCls, text, mask, targets, Nonan_id):
        ImageCls = ImageCls.unsqueeze(1).expand(-1, text.shape[1], -1, -1, -1)
        # (4, 171, 512, 24, 24)

        textB = text.squeeze(2)
        # (4, 171, 512)
        imageB = torch.mul(ImageCls , mask)


        index = [
            torch.unique(targets[i]).long() if torch.unique(targets[i]).long()[-1].item() != 255
            else torch.unique(targets[i])[:-1].long()
            for i in range(targets.shape[0])
        ]
        # self.caculateNan(index, mask.sum(dim=(2,3,4)))

        loss = 0
        diagonal_tensor = torch.eye(text.shape[1], device=text.device)
        # mask=mask.sum(dim=(2, 3, 4))
        # print(f"mask{mask}")
        for i in  Nonan_id:
            # print(f"i{i}")
            # print(f"index_i{index[i]}")
            # print(f"mask_i{mask[i, index[i]]}")

            textB_i = textB[i]
            imageB_i = imageB[i, index[i]]
            mask_i = mask[i, index[i]]

            imageAvg_i = (imageB_i).sum(dim=(2, 3)) / mask_i.sum(dim=(2, 3))
            # ( len(index[i], 512, 24, 24) -> ( len(index[i], 512)
            cosine_i = F.cosine_similarity(imageAvg_i.unsqueeze(1), textB_i.unsqueeze(0), dim=2)
            diagonal_i = diagonal_tensor[index[i]]
            loss_i = F.binary_cross_entropy_with_logits(cosine_i, diagonal_i)

            mask_class = (mask[i].sum(dim=(1,2,3)) > 0.1)
            a = imageB_i.sum(dim=(1, 2, 3)).detach().cpu().numpy()
            c = mask_i.sum(dim=(1, 2, 3)).detach().cpu().numpy()
            #
            # print(f"mask_i{c}")
            # print(f"index{index[i]}")
            # print(f"mask_all{mask[i].sum(dim=(1,2,3))}")

            loss += loss_i

        if len(Nonan_id) != 0:
            loss = loss/len(Nonan_id)

        return loss/2

    def caculateLoss2(self, ImageCls, ImageNoCls):
        loss = torch.einsum('ijkl,ijkl->i', ImageCls, ImageNoCls)
        loss = loss.sum(dim=0)
        loss = loss*loss
        return loss


    def caculateLoss3(self, image, imageB):
        loss = self.mse_loss(image, imageB)
        return loss


    def AddNoiseToimage(self, noise, text, image, targets,vis_guidance):
        """
        Args:
            Arguments:
                noise (4, 171, 1, 512)
                text (4, 171, 1, 512) (B, T, P, C)
                image (4, 512, 24, 24) (B, C, H, W)
                targets (4, 384, 384) (B, H, W)
            Returns:
                image (4, 512, 24, 24)
                loss (1,)
        """

        noise, mask, Nonan_id = self.NoiseToImage(image, noise, targets)

        guidance_noises = [ conv(noise) for conv in self.conv ]
        vis_guidance = [ torch.add(vis,guidance_noise) for vis,guidance_noise in zip(vis_guidance, guidance_noises) ]

        # ImageCls = self.GetImageCls(image)
        # ImageNoCls = self.GetImageNoCls(image)
        # (4, 512, 24, 24)


        loss1= 0
        # loss2 = self.caculateLoss2(ImageCls, ImageNoCls)


        image = torch.add(image , noise)
        # loss1 = self.caculateLoss1(image, text, mask, targets, Nonan_id)


        # imageB = torch.cat((ImageCls, ImageNoCls), dim=1)
        # imageB = self.conv4(image)

        # loss3 = self.caculateLoss3(image, imageB)
        # print(f"loss1 {loss1}")
        # print(f"loss2 {loss2}")
        # print(f"loss3 {loss3}")

        return  image,vis_guidance,loss1

    def AddNoiseTotext(self, noise, text):
        return torch.add(noise, text)

    def correlation(self, img_feats, text_feats):
        # img_feats = F.normalize(img_feats, dim=1) # B C H W
        # text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)

        """
        
        for b in range(img_feats.shape[0]):
            for p in range(text_feats.shape[2]):
                for t in range(text_feats.shape[1]):
                    for h in range(img_fears.shape[2]):
                       for w in range(img_feats.shape[3]):
                         for c in range(text_feats.shape[4]):
                           corr[b, p, t, h, w] += img_feats[b, c, h, w]*text_feats[b, t, p, c]
                          
        """


        return corr

    def getnoise(self, shape, device):
        noise = torch.normal(self.mean, self.stddev, shape, device=device,dtype=torch.float32)
        return  noise

    def forward(self,  image_features, text_features, targets, vis_guidance):
        """
        h * w
        Args:
            Arguments:
                img_feats: (B, C, H, W)
                                512 24 24
                text_feats: (B, T, P, C)
                               171 1  512


        Returns:
        """


        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        # print(image_features.device)
        # print(torch.cuda.device_count())  # 可用的 GPU 数量
        # print(torch.cuda.current_device())  # 当前使用的 GPU
        # print(torch.cuda.get_device_name(0))  # 当前 GPU 的名称
        #
        # self.times += 1
        # self.sumImage += image_features
        # self.sumText += text_features
        # self.maxnI = max(torch.max(image_features).item(), self.maxnI)
        # self.maxnT = max(torch.max(text_features).item(), self.maxnT)
        # self.minnI = min(torch.min(image_features).item(), self.minnI)
        # self.minnT = min(torch.min(text_features).item(), self.minnT)
        #
        #
        # print(f"imagemax: {self.maxnI} min: {self.minnI} mean {self.sumImage.mean().item()/self.times} ")
        # print(f"textmax: {self.maxnT} min: {self.minnT} mean {self.sumText.mean().item()/self.times} ")
        # print("\n")

        # Ori_corr = self.correlation(image_features, text_features)
        #

        text_noise = self.getnoise(text_features.shape, image_features.device) # B T P C

        image_features,vis_guidance,loss = self.AddNoiseToimage(text_noise, text_features, image_features, targets,vis_guidance)
        image_features = F.normalize(image_features, dim=1)

        text_features  = self.AddNoiseTotext(text_noise,  text_features)
        text_features = F.normalize(text_features, dim=1)



        # Now_corr = self.correlation(image_features,text_features)

        # loss4 = self.mse_loss(Ori_corr, Now_corr)
        # print(f"loss4: {loss4}")
        # print(f"text_noise: {text_noise}")
        # print(f"image_features: {image_features}")
        # print(f"text_features: {text_features}")


        return image_features, text_features, vis_guidance,loss
















