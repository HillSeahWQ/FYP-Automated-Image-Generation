import os
import torch
from torchvision import transforms, datasets
from torchvision.models import inception_v3
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception_score import InceptionScore
from PIL import Image

def load_images(image_dir, transform, batch_size=32):
    """Loads images from a directory using a PyTorch DataLoader."""
    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def calculate_fid_is(image_dir, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Calculates FID and IS for images in a given directory.
    
    Parameters:
        image_dir (str): Path to the image directory.
        device (str): Device to perform calculations ('cuda' or 'cpu').
    """
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Required input size for InceptionV3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])
    
    # Load images
    dataloader = load_images(image_dir, transform)
    
    # Initialize metrics
    fid = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore().to(device)
    
    # Load pre-trained InceptionV3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # Loop through images and compute embeddings
    for batch, _ in dataloader:
        batch = batch.to(device)
        
        with torch.no_grad():
            embeddings = inception_model(batch)  # Get embeddings
        
        fid.update(batch, real=True)  # Update FID with generated images
        is_metric.update(batch)      # Update IS
    
    # Compute metrics
    fid_score = fid.compute()
    is_mean, is_std = is_metric.compute()
    
    print(f"FID Score: {fid_score.item():.4f}")
    print(f"Inception Score: {is_mean.item():.4f} ± {is_std.item():.4f}")

# Directory containing the images
image_directory = "/path/to/your/image/directory"

if __name__ == "__main__":
    # Calculate FID and IS
    #calculate_fid_is(image_directory)
    print("done")
