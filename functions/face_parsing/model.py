#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import PIL
import numpy as np
from scipy.linalg import sqrtm

from .resnet import Resnet18


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
    

class FaceParseTool(nn.Module):
    def __init__(self, n_classes=19, ref_path=None):
        super(FaceParseTool, self).__init__()
        self.n_classes = n_classes
        self.net = BiSeNet(self.n_classes)
        self.net = self.net.cuda()
        self.net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '79999_iter.pth'))) # changed from "/workspace/ddgm/functions/face_parsing/79999_iter.pth"
        self.net.eval()
        
        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
#         self.reference_img_path = "/userhome/yjw/ddgm_exp/functions/face_parsing/00234.png" 
        self.reference_img_path = "/workspace/ddgm/functions/face_parsing/43.jpg" if not ref_path else ref_path
        img = PIL.Image.open(self.reference_img_path).convert("RGB")
        # preprocess for ref image
        image = img.resize((512, 512), PIL.Image.BILINEAR)
        img = self.to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        self.ref = img
        
        self.preprocess = torchvision.transforms.Normalize( # preprocessing for x0|t
            (0.485*2-1, 0.456*2-1, 0.406*2-1), 
            (0.229*2, 0.224*2, 0.225*2)
        )
    

    def get_residual(self, image):
        image = torch.nn.functional.interpolate(image, size=512, mode='bicubic')
        image = self.preprocess(image)
        
        ref_mask = self.net(self.ref)[0]
        img_mask = self.net(image)[0]
        
        return ref_mask - img_mask
    
    def get_residual_y(self, image, image_y):
        image = torch.nn.functional.interpolate(image, size=512, mode='bicubic')
        image = self.preprocess(image)

        image_y = torch.nn.functional.interpolate(image_y, size=512, mode='bicubic')
        image_y = self.preprocess(image_y)

        ref_mask = self.net(image_y)[0]
        img_mask = self.net(image)[0]
        
        return ref_mask - img_mask
    
    def get_mask(self, image, id_num):
        image = torch.nn.functional.interpolate(image, size=512, mode='bicubic')
        image = self.preprocess(image)
        img_mask = self.net(image)[0]
        img_mask = img_mask.squeeze(0).cpu().detach().numpy().argmax(0)
        img_mask = torch.Tensor(img_mask)
        if type(id_num) == list:
            img_res = (img_mask == id_num[0]) * 1
            for i in range(1, len(id_num)):
                img_res += (img_mask == id_num[i]) * 1
            img_mask = img_res
        else:
            img_mask = (img_mask == id_num) * 1
        img_mask = img_mask.reshape(1, 1, 512, 512).float()
        img_mask = torch.nn.functional.interpolate(img_mask, size=256, mode='bicubic')
        return img_mask.cuda().detach()
    
    def save_segmentation_map(self, input_image_path, output_image_path):
        # Load and preprocess the input image
        img = PIL.Image.open(input_image_path).convert("RGB").resize((512, 512), PIL.Image.BILINEAR)
        img_tensor = self.to_tensor(img).unsqueeze(0).cuda()
        
        # Perform inference to get the segmentation map
        with torch.no_grad():
            output = self.net(img_tensor)[0]
        
        # Get the segmentation map as a numpy array
        segmentation_map = output.squeeze().cpu().numpy().argmax(0)
        
        # Convert the segmentation map to a color image for better visualization
        colormap = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
        for label in range(self.n_classes):
            colormap[segmentation_map == label] = [label * 10 % 256, label * 20 % 256, label * 30 % 256]  # Customize colors
        
        # Save the image
        seg_image = PIL.Image.fromarray(colormap)
        seg_image.save(output_image_path)
        print(f"Segmentation map saved to {output_image_path}")
    
    def normalize_mask(self, mask, method="min-max"):
        if method == "min-max":
            # Min-Max normalization (calculate min and max across spatial dimensions)
            mask_min = mask.view(mask.size(0), -1).min(dim=1, keepdim=True).values  # Flatten spatial dims
            mask_min = mask_min.view(mask.size(0), 1, 1, 1)  # Reshape for broadcasting

            mask_max = mask.view(mask.size(0), -1).max(dim=1, keepdim=True).values  # Flatten spatial dims
            mask_max = mask_max.view(mask.size(0), 1, 1, 1)  # Reshape for broadcasting

            normalized_mask = (mask - mask_min) / (mask_max - mask_min + 1e-8)

        elif method == "softmax":
            # Softmax normalization across channels
            normalized_mask = torch.softmax(mask, dim=1)

        elif method == "l2":
            # L2 normalization
            norm = torch.norm(mask.view(mask.size(0), -1), dim=1, keepdim=True) + 1e-8
            norm = norm.view(mask.size(0), 1, 1, 1)  # Reshape for broadcasting
            normalized_mask = mask / norm

        else:
            raise ValueError("Unsupported normalization method. Choose from 'min-max', 'softmax', or 'l2'.")

        return normalized_mask

    def get_gaussian_kernel(self, image, sigma, normalization="min-max"):
        image = torch.nn.functional.interpolate(image, size=512, mode='bicubic')
        image = self.preprocess(image)
        
        ref_mask = self.net(self.ref)[0]
        img_mask = self.net(image)[0]
        
        # Normalize the masks
        ref_mask = self.normalize_mask(ref_mask, method=normalization)
        img_mask = self.normalize_mask(img_mask, method=normalization)

        # Compute pixel-wise distance (squared L2 norm)
        mask_distance = torch.mean((img_mask - ref_mask) ** 2, dim=(1, 2, 3))  # Per-image distance

        # Apply Gaussian function
        gaussian_similarity = torch.exp(-mask_distance / (2 * sigma ** 2))  # Gaussian kernel

        return gaussian_similarity

        
    def calculate_mask_distance(self, image_path1, image_path2):
        # Load and preprocess the first image
        img1 = PIL.Image.open(image_path1).convert('RGB')
        img1 = img1.resize((512, 512), PIL.Image.BILINEAR)
        img1 = self.to_tensor(img1).unsqueeze(0).cuda()

        # Load and preprocess the second image
        img2 = PIL.Image.open(image_path2).convert('RGB')
        img2 = img2.resize((512, 512), PIL.Image.BILINEAR)
        img2 = self.to_tensor(img2).unsqueeze(0).cuda()

        # Get segmentation masks
        mask1 = self.net(img1)[0]
        mask2 = self.net(img2)[0]

        # Calculate the distance between masks (e.g., L2 norm)
        distance = torch.norm(mask1 - mask2, dim=1).mean()
        return distance
    

    def calculate_fid(self, image_path1, image_path2):
        # load and preprocess the images
        img1 = PIL.Image.open(image_path1).convert('RGB')
        img1 = img1.resize((512, 512), PIL.Image.BILINEAR)
        img1 = self.to_tensor(img1).unsqueeze(0).cuda()

        img2 = PIL.Image.open(image_path2).convert('RGB')
        img2 = img2.resize((512, 512), PIL.Image.BILINEAR)
        img2 = self.to_tensor(img2).unsqueeze(0).cuda()

        # get segmentation masks
        mask1 = self.net(img1)[0].detach().cpu().numpy()
        mask2 = self.net(img2)[0].detach().cpu().numpy()

        # calculate means and covariances
        mu1 = mask1.mean(axis=(1, 2))
        sigma1 = np.cov(mask1.reshape(mask1.shape[0], -1), rowvar=False)

        mu2 = mask2.mean(axis=(1, 2))
        sigma2 = np.cov(mask2.reshape(mask2.shape[0], -1), rowvar=False)

        # ensure covariance matrices are valid
        if sigma1.ndim < 2:
            sigma1 = np.atleast_2d(sigma1)
        if sigma2.ndim < 2:
            sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # ensure sigma1 and sigma2 are matrices
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        covmean, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return np.dot(diff, diff.T).item() + np.trace(sigma1 + sigma2 - 2 * covmean)
    

    def get_polynomial_kernel(self, image, degree=2, alpha=1.0, c=0.0, normalization="min-max"):
        image = torch.nn.functional.interpolate(image, size=512, mode='bicubic')
        image = self.preprocess(image)
        
        ref_mask = self.net(self.ref)[0]
        img_mask = self.net(image)[0]
        
        # Normalize the masks
        ref_mask = self.normalize_mask(ref_mask, method=normalization)
        img_mask = self.normalize_mask(img_mask, method=normalization)

        dot_product = torch.sum(ref_mask * img_mask, dim=(1, 2, 3))  # Sum over spatial dimensions

        polynomial_kernel = (alpha * dot_product + c) ** degree

        return polynomial_kernel

    def get_sigmoid_kernel(self, image, alpha=1.0, c=0.0, normalization="min-max"):
        image = torch.nn.functional.interpolate(image, size=512, mode='bicubic')
        image = self.preprocess(image)

        ref_mask = self.net(self.ref)[0]
        img_mask = self.net(image)[0]

        # Normalize the masks
        ref_mask = self.normalize_mask(ref_mask, method=normalization)
        img_mask = self.normalize_mask(img_mask, method=normalization)

        dot_product = torch.sum(ref_mask * img_mask, dim=(1, 2, 3))  # Sum over spatial dimensions

        # Apply sigmoid kernel transformation
        sigmoid_kernel = torch.tanh(alpha * dot_product + c)

        return sigmoid_kernel
    
    def get_euclidean_distance(self, image, normalization="min-max"):
        image = torch.nn.functional.interpolate(image, size=512, mode='bicubic')
        image = self.preprocess(image)

        ref_mask = self.net(self.ref)[0]
        img_mask = self.net(image)[0]

        # Normalize masks
        ref_mask = self.normalize_mask(ref_mask, method=normalization)
        img_mask = self.normalize_mask(img_mask, method=normalization)

        # Compute Euclidean distance
        euclidean_distance = torch.norm(ref_mask - img_mask, p=2, dim=(1, 2, 3))

        return euclidean_distance


    def get_cosine_similarity(self, image, normalization="l2"):
        image = torch.nn.functional.interpolate(image, size=512, mode='bicubic')
        image = self.preprocess(image)

        ref_mask = self.net(self.ref)[0]
        img_mask = self.net(image)[0]

        # Normalize masks using L2 norm (best for cosine similarity)
        ref_mask = self.normalize_mask(ref_mask, method=normalization)
        img_mask = self.normalize_mask(img_mask, method=normalization)

        # Compute Cosine similarity
        dot_product = torch.sum(ref_mask * img_mask, dim=(1, 2, 3))
        norm_ref = torch.norm(ref_mask, p=2, dim=(1, 2, 3))
        norm_img = torch.norm(img_mask, p=2, dim=(1, 2, 3))

        cosine_similarity = dot_product / (norm_ref * norm_img + 1e-8)  # Avoid division by zero

        return cosine_similarity


    def get_pearson_correlation(self, image, normalization="min-max"):
        image = torch.nn.functional.interpolate(image, size=512, mode='bicubic')
        image = self.preprocess(image)

        ref_mask = self.net(self.ref)[0]
        img_mask = self.net(image)[0]

        # Normalize masks
        ref_mask = self.normalize_mask(ref_mask, method=normalization)
        img_mask = self.normalize_mask(img_mask, method=normalization)

        # Compute Pearson correlation
        ref_mean = torch.mean(ref_mask, dim=(1, 2, 3), keepdim=True)
        img_mean = torch.mean(img_mask, dim=(1, 2, 3), keepdim=True)

        ref_centered = ref_mask - ref_mean
        img_centered = img_mask - img_mean

        numerator = torch.sum(ref_centered * img_centered, dim=(1, 2, 3))
        denominator = torch.sqrt(torch.sum(ref_centered ** 2, dim=(1, 2, 3)) * torch.sum(img_centered ** 2, dim=(1, 2, 3)) + 1e-8)

        pearson_correlation = numerator / denominator

        return pearson_correlation


if __name__ == "__main__":
    net = BiSeNet(19)
    net.cuda()
    net.load_state_dict(torch.load("/workspace/ddgm/functions/face_parsing/79999_iter.pth"))
    net.eval()
    # reference_img_path = "/workspace/ddgm/exp/datasets/celeba_hq/celeba/00618.png"
    reference_img_path = "/workspace/TediGAN/ext/experiment/images/img/input_img.jpg"
    from PIL import Image
    img = Image.open(reference_img_path)
    image = img.resize((512, 512), Image.BILINEAR)
    to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    ref = img
    ref_mask = net(ref)[0]
    print(ref_mask.size())
    # exit(0)
#     print(ref_mask[0, 0])
    ref_mask = ref_mask.squeeze(0).cpu().detach().numpy().argmax(0)
#     print(ref_mask)
#     print(len(ref_mask), len(ref_mask[0]))
    import numpy as np
    import cv2
    ref_mask = np.array(ref_mask)
    for i in range(19):
        mask = (ref_mask == i) * 255.
        mask = mask.astype(np.uint8)
        cv2.imwrite("./mask_{}.png".format(i), mask)
    

# 1: skin
# 12&13: lip
# 16: cloth
# 17: hair