# import os
# import torch
# from torchvision import transforms, datasets
# from torchvision.models import inception_v3
# from torchmetrics.image.fid import FrechetInceptionDistance
# from torchmetrics.image.inception_score import InceptionScore
# from PIL import Image

# def load_images(image_dir, transform, batch_size=32):
#     """Loads images from a directory using a PyTorch DataLoader."""
#     dataset = datasets.ImageFolder(root=image_dir, transform=transform)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     return dataloader

from pathlib import Path
import os
from tqdm import tqdm
from functions.clip.base_clip import CLIPEncoder
from functions.face_parsing.model import FaceParseTool
from functions.landmark.model import FaceLandMarkTool
from functions.arcface.model import IDLoss
from functions.anime2sketch.model import FaceSketchTool

def get_image_paths(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    image_paths = []
    for root, _, files in os.walk(directory):  # os.walk traverses the directory
        for file in files:
            if file.lower().endswith(extensions):  # Check file extension
                image_paths.append(os.path.join(root, file))  # Join root and file name
    return image_paths

# To edit
gen_image_directory = Path.cwd() / "exp" / "res" / "clip_parse_ref294jpg" / "old"
ref_image_path = r"C:\Users\cliff\Hill\FYP-Automated-Image-Generation\Face-GD\images\id10.png"
text_cond = "black woman"

conditions = {
    "clip": [],
    "parse": [],
    "landmark": [],
    "arc": [],
    "sketch": []
}

img_paths = get_image_paths(gen_image_directory)
clip_encoder = CLIPEncoder().cuda()
parser = FaceParseTool(ref_path=ref_image_path).cuda()
img2sketch = FaceSketchTool(ref_path=ref_image_path).cuda()
img2landmark = FaceLandMarkTool(ref_path=ref_image_path).cuda()
idloss = IDLoss(ref_path=ref_image_path).cuda()


for img_path in tqdm(img_paths):

    if "clip" in list(conditions.keys()):
        conditions["clip"].append(clip_encoder.calculate_euclidean_distance(image_path=img_path, text=text_cond))

    if "parse" in list(conditions.keys()):
        conditions["parse"].append(parser.calculate_mask_distance(ref_image_path, img_path))

    if "sketch" in list(conditions.keys()):
        conditions["sketch"].append(img2sketch.calculate_sketch_distance(ref_image_path, img_path))

    if "landmark" in list(conditions.keys()):
        conditions["landmark"].append(img2landmark.calculate_landmark_distance(ref_image_path, img_path))

    if "arc" in list(conditions.keys()):
        conditions["arc"].append(idloss.calculate_id_distance(ref_image_path, img_path))


for key, val in conditions.items():
    print(f"Mean {key} distance = {sum(val)/len(img_paths)}")






# def calculate_fid_is(image_dir, device="cuda" if torch.cuda.is_available() else "cpu"):
#     """
#     Calculates FID and IS for images in a given directory.
    
#     Parameters:
#         image_dir (str): Path to the image directory.
#         device (str): Device to perform calculations ('cuda' or 'cpu').
#     """
#     # Define image transformations
#     transform = transforms.Compose([
#         transforms.Resize((299, 299)),  # Required input size for InceptionV3
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
#     ])
    
#     # Load images
#     dataloader = load_images(image_dir, transform)
    
#     # Initialize metrics
#     fid = FrechetInceptionDistance(feature=2048).to(device)
#     is_metric = InceptionScore().to(device)
    
#     # Load pre-trained InceptionV3 model
#     inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
#     inception_model.eval()
    
#     # Loop through images and compute embeddings
#     for batch, _ in dataloader:
#         batch = batch.to(device)
        
#         with torch.no_grad():
#             embeddings = inception_model(batch)  # Get embeddings
        
#         fid.update(batch, real=True)  # Update FID with generated images
#         is_metric.update(batch)      # Update IS
    
#     # Compute metrics
#     fid_score = fid.compute()
#     is_mean, is_std = is_metric.compute()
    
#     print(f"FID Score: {fid_score.item():.4f}")
#     print(f"Inception Score: {is_mean.item():.4f} ± {is_std.item():.4f}")

# # Directory containing the images
# image_directory = "/path/to/your/image/directory"

# if __name__ == "__main__":
#     # Calculate FID and IS
#     #calculate_fid_is(image_directory)
#     print("done")

# from pytorch_fid import fid_score

# # Paths to real and generated image directories
# real_images_path = "path_to_real_images"
# model1_generated_path = "path_to_model1_images"
# model2_generated_path = "path_to_model2_images"

# # Compute FID for both models
# fid_model1 = fid_score.calculate_fid_given_paths([real_images_path, model1_generated_path], batch_size=50, device='cuda', dims=2048)
# fid_model2 = fid_score.calculate_fid_given_paths([real_images_path, model2_generated_path], batch_size=50, device='cuda', dims=2048)

# print(f"FID Score Model 1: {fid_model1}")
# print(f"FID Score Model 2: {fid_model2}")
