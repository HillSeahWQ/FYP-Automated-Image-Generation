import os
import torch
import torch.nn as nn
from .models.mobilefacenet import MobileFaceNet
import glob
import cv2
from PIL import Image
from .Retinaface import Retinaface
from .common.utils import BBox


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
