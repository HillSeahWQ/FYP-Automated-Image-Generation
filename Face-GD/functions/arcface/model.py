import os
import torch
from torch import nn
from .facial_recognition.model_irse import Backbone
import torchvision
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm


class IDLoss(nn.Module):

    def __init__(self, ref_path=None):
        super(IDLoss, self).__init__()
#         print('Loading ResNet ArcFace for ID Loss')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_ir_se50.pth"))) # changed from "/workspace/ddgm/functions/arcface/model_ir_se50.pth"
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

        self.to_tensor = torchvision.transforms.ToTensor()

        self.ref_path = "/workspace/ddgm/functions/arcface/land.png" if not ref_path else ref_path
        
        img = Image.open(self.ref_path)
        image = img.resize((256, 256), Image.BILINEAR)
        img = self.to_tensor(image)
        img = img * 2 - 1
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        self.ref = img


    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats
    

    def get_residual(self, image):
        img_feat = self.extract_feats(image)
        ref_feat = self.extract_feats(self.ref)
        return ref_feat - img_feat
    
    
    def get_gaussian_kernel(self, image, sigma):
        img_feat = self.extract_feats(image)
        ref_feat = self.extract_feats(self.ref)

        distance = torch.norm(img_feat - ref_feat, dim=-1)

        # Apply Gaussian kernel
        gaussian_similarity = torch.exp(-distance**2 / (2 * sigma**2))
        
        return gaussian_similarity
    
    
    def calculate_id_distance(self, image_path1, image_path2):

        # load and preprocess the images
        img1 = Image.open(image_path1).resize((256, 256), Image.BILINEAR)
        img1 = self.to_tensor(img1)
        img1 = img1 * 2 - 1
        img1 = torch.unsqueeze(img1, 0).cuda()

        img2 = Image.open(image_path2).resize((256, 256), Image.BILINEAR)
        img2 = self.to_tensor(img2)
        img2 = img2 * 2 - 1
        img2 = torch.unsqueeze(img2, 0).cuda()

        # extract features
        feats1 = self.extract_feats(img1)
        feats2 = self.extract_feats(img2)

        # calculate the distance
        distance = torch.norm(feats1 - feats2, dim=1).mean()
        return distance
    

    def calculate_fid(self, image_path1, image_path2):


        def preprocess_image(image_path):
            img = Image.open(image_path).resize((256, 256), Image.BILINEAR)
            img = self.to_tensor(img)
            img = img * 2 - 1
            img = torch.unsqueeze(img, 0).cuda()
            return self.extract_feats(img).detach().cpu().numpy()

        # extract face ID embeddings for both images
        feats1 = preprocess_image(image_path1)
        feats2 = preprocess_image(image_path2)

        # calculate mean and covariance
        mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
        mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

        # ensure covariance matrices are valid
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        # compute FID

        diff = mu1 - mu2

        # ensure sigma1 and sigma2 are matrices
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        covmean, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid_score = np.dot(diff, diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid_score





