import os
import torch
import torch.nn as nn
from .models.mobilefacenet import MobileFaceNet
import glob
import cv2
from PIL import Image
from .Retinaface import Retinaface
from .common.utils import BBox
import numpy as np
from scipy.linalg import sqrtm


class FaceLandMarkTool(nn.Module):
    def __init__(self, ref_path=None):
        super(FaceLandMarkTool, self).__init__()
        self.out_size = 112
        map_location = lambda storage, loc: storage.cuda()
        self.landmark_net = MobileFaceNet([self.out_size, self.out_size], 136)
        checkpoint = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint", "mobilefacenet_model_best.pth.tar"), map_location=map_location) # changed from '/workspace/ddgm/functions/landmark/checkpoint/mobilefacenet_model_best.pth.tar'
        self.landmark_net.load_state_dict(checkpoint['state_dict'])
        self.landmark_net = self.landmark_net.eval()
        
        self.ref_path = "/workspace/ddgm/functions/landmark/3650.png" if not ref_path else ref_path
        img = cv2.imread(self.ref_path)
        img = cv2.resize(img, (256, 256))

        retinaface = Retinaface.Retinaface()
        faces = retinaface(img)
        face = faces[0]
        
        x1 = face[0]
        y1 = face[1]
        x2 = face[2]
        y2 = face[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h])*1.2)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        
        self.top, self.bottom, self.left, self.right = new_bbox.top, new_bbox.bottom, new_bbox.left, new_bbox.right
        
        cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
        cropped_face = cv2.resize(cropped, (self.out_size, self.out_size))
        
        test_face = cropped_face.copy()
        test_face = test_face/255.0
        test_face = test_face.transpose((2, 0, 1))
        test_face = test_face.reshape((1,) + test_face.shape)

        input_ref = torch.from_numpy(test_face).float()
        input_ref = torch.autograd.Variable(input_ref)

        self.landmark_ref = self.landmark_net(input_ref)[0].cuda()
    

    def get_residual(self, image):
        image = (image + 1.0) / 2.0
        image = image[:, :, self.top:self.bottom, self.left:self.right]
        # print(self.top, self.bottom)
        image = torch.nn.functional.interpolate(image, size=self.out_size, mode='bicubic')
        landmark_img = self.landmark_net(image)[0]
        return self.landmark_ref - landmark_img
    
    
    def save_landmarks(self, output_path):
        from PIL import Image, ImageDraw
        # Convert the tensor to numpy for processing
        landmarks = self.landmark_ref.cpu().detach().numpy()
        
        # Reshape the landmarks to (68, 2) format for easier plotting
        landmarks = landmarks.reshape(-1, 2)  # 68 points, each with (x, y)

        # Load the reference image again to overlay the landmarks
        img = cv2.imread(self.ref_path)
        img = cv2.resize(img, (256, 256))

        # Convert the image from BGR (OpenCV format) to RGB (Pillow format)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to a Pillow Image
        pil_img = Image.fromarray(img)

        # Create a drawing context
        draw = ImageDraw.Draw(pil_img)

        # Adjust landmarks to the original scale of the image
        for (x, y) in landmarks:
            # Rescale to match the bounding box and image dimensions
            x = int((x * (self.right - self.left) / self.out_size) + self.left)
            y = int((y * (self.bottom - self.top) / self.out_size) + self.top)

            # Ensure that the coordinates are within the image bounds
            x = max(0, min(x, pil_img.width - 1))
            y = max(0, min(y, pil_img.height - 1))

            # Draw a small circle (ellipse) at each landmark
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(0, 255, 0))  # Green circle for landmarks

        # Save the image with landmarks overlaid using Pillow
        pil_img.save(output_path)
        print(f"Saved image with landmarks to {output_path}")

    
    def normalize_landmarks(self, landmarks, image_width, image_height):
        # Reshape if necessary
        if landmarks.dim() == 2 and landmarks.size(1) == 136:  # Check if it's in [1, 136] format
            landmarks = landmarks.view(landmarks.shape[0], -1, 2)  # Reshape to [batch_size, num_landmarks, 2]
        
        if landmarks.dim() == 3:
            landmarks[:, :, 0] /= image_width  # Normalize x-coordinates
            landmarks[:, :, 1] /= image_height  # Normalize y-coordinates
        else:
            raise ValueError(f"Unexpected tensor shape: {landmarks.shape}")

        return landmarks
    
    def get_gaussian_kernel(self, image, sigma):
        image = (image + 1.0) / 2.0
        image = image[:, :, self.top:self.bottom, self.left:self.right]
        image = torch.nn.functional.interpolate(image, size=self.out_size, mode='bicubic')

        landmark_img = self.landmark_net(image)[0]

        # image_width = self.out_size
        # image_height = self.out_size
        # landmark_img = self.normalize_landmarks(landmark_img, image_width, image_height)
        # ref_landmarks = self.normalize_landmarks(self.landmark_ref, image_width, image_height)
        # landmark_distance = torch.norm(landmark_img - ref_landmarks, dim=-1)  # (batch_size, num_landmarks)
        # mean_distance = torch.mean(landmark_distance, dim=1)  # Mean distance per image (batch_size,)
        
        landmark_img = landmark_img.view(landmark_img.shape[0], -1, 2)  # Reshape to [batch_size, num_landmarks, 2]
        landmarks_ref = self.landmark_ref.view(self.landmark_ref.shape[0], -1, 2)  # Reshape to [batch_size, num_landmarks, 2]
        landmark_distance = torch.norm(landmark_img - landmarks_ref, dim=-1)  # (batch_size, num_landmarks)
        mean_distance = torch.mean(landmark_distance, dim=1)  # Mean distance per image (batch_size,)

        # Apply Gaussian kernel to the mean distance
        gaussian_similarity = torch.exp(-mean_distance / (2 * sigma ** 2))  # Gaussian kernel similarity
        return gaussian_similarity
    
    def calculate_landmark_distance(self, image_path1, image_path2):

        # preprocess the images
        img1 = cv2.imread(image_path1)
        img1 = cv2.resize(img1, (256, 256))

        retinaface = Retinaface.Retinaface()
        faces1 = retinaface(img1)
        face1 = faces1[0]

        x1 = face1[0]
        y1 = face1[1]
        x2 = face1[2]
        y2 = face1[3]
        w1 = x2 - x1 + 1
        h1 = y2 - y1 + 1
        size1 = int(min([w1, h1]) * 1.2)
        cx1 = x1 + w1 // 2
        cy1 = y1 + h1 // 2
        x1 = cx1 - size1 // 2
        x2 = x1 + size1
        y1 = cy1 - size1 // 2
        y2 = y1 + size1

        new_bbox1 = list(map(int, [x1, x2, y1, y2]))
        new_bbox1 = BBox(new_bbox1)

        cropped1 = img1[new_bbox1.top:new_bbox1.bottom, new_bbox1.left:new_bbox1.right]
        cropped_face1 = cv2.resize(cropped1, (self.out_size, self.out_size))

        test_face1 = cropped_face1.copy()
        test_face1 = test_face1 / 255.0
        test_face1 = test_face1.transpose((2, 0, 1))
        test_face1 = test_face1.reshape((1,) + test_face1.shape)

        input_img1 = torch.from_numpy(test_face1).float().cuda()  # Move to GPU

        landmark1 = self.landmark_net(input_img1)[0]

        img2 = cv2.imread(image_path2)
        img2 = cv2.resize(img2, (256, 256))

        faces2 = retinaface(img2)
        face2 = faces2[0]

        x1 = face2[0]
        y1 = face2[1]
        x2 = face2[2]
        y2 = face2[3]
        w2 = x2 - x1 + 1
        h2 = y2 - y1 + 1
        size2 = int(min([w2, h2]) * 1.2)
        cx2 = x1 + w2 // 2
        cy2 = y1 + h2 // 2
        x1 = cx2 - size2 // 2
        x2 = x1 + size2
        y1 = cy2 - size2 // 2
        y2 = y1 + size2

        new_bbox2 = list(map(int, [x1, x2, y1, y2]))
        new_bbox2 = BBox(new_bbox2)

        cropped2 = img2[new_bbox2.top:new_bbox2.bottom, new_bbox2.left:new_bbox2.right]
        cropped_face2 = cv2.resize(cropped2, (self.out_size, self.out_size))

        test_face2 = cropped_face2.copy()
        test_face2 = test_face2 / 255.0
        test_face2 = test_face2.transpose((2, 0, 1))
        test_face2 = test_face2.reshape((1,) + test_face2.shape)

        input_img2 = torch.from_numpy(test_face2).float().cuda()  # Move to GPU

        landmark2 = self.landmark_net(input_img2)[0]

        # calculate the distance between the landmarks
        distance = torch.norm(landmark1 - landmark2, dim=0).mean()
        return distance
    
    def calculate_fid(self, image_path1, image_path2):

        def preprocess_image(image_path):
            img = cv2.imread(image_path)
            img = cv2.resize(img, (256, 256))

            retinaface = Retinaface.Retinaface()
            faces = retinaface(img)
            face = faces[0]

            x1 = face[0]
            y1 = face[1]
            x2 = face[2]
            y2 = face[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(min([w, h]) * 1.2)
            cx = x1 + w // 2
            cy = y1 + h // 2
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)

            cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
            cropped_face = cv2.resize(cropped, (self.out_size, self.out_size))

            test_face = cropped_face.copy()
            test_face = test_face / 255.0
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)

            input_img = torch.from_numpy(test_face).float().cuda()
            return self.landmark_net(input_img)[0].detach().cpu().numpy()

        # extract features for both images
        landmark1 = preprocess_image(image_path1)
        landmark2 = preprocess_image(image_path2)

        # calculate mean and covariance
        mu1, sigma1 = landmark1.mean(axis=0), np.cov(landmark1, rowvar=False)
        mu2, sigma2 = landmark2.mean(axis=0), np.cov(landmark2, rowvar=False)

        # ensure covariance matrices are valid
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        # compute FID
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid_score = np.dot(diff, diff).item() + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid_score
