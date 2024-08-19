import cv2
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os

# Load the pretrained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Path to the input and output folders
input_folder = 'input_images/'
output_folder = 'output_mask_image/'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Function to create a binary garment mask using Mask R-CNN
def create_binary_garment_mask(image, prediction):
    masks = prediction['masks']
    scores = prediction['scores']
    labels = prediction['labels']
    
    threshold = 0.5
    valid_masks = masks[scores > threshold]
    valid_labels = labels[scores > threshold]

    if valid_masks.shape[0] > 0:
        garment_mask = valid_masks[0, 0].mul(255).byte().cpu().numpy()
        _, binary_mask = cv2.threshold(garment_mask, 127, 255, cv2.THRESH_BINARY)
        return binary_mask
    else:
        return np.zeros((image.height, image.width), dtype=np.uint8)

# Custom HSV bounds based on filename
def get_hsv_bounds(filename):
    if "image_1" in filename:
        lower_bound = np.array([40, 0, 92])
        upper_bound = np.array([119, 216, 239])
    elif "image_2" in filename:
        lower_bound = np.array([20, 15, 14])
        upper_bound = np.array([173, 255, 255])
    elif "image_3" in filename:
        lower_bound = np.array([0, 0, 9])
        upper_bound = np.array([21, 32, 91])
    elif "image_4" in filename:
        lower_bound = np.array([15, 63, 51])
        upper_bound = np.array([47, 255, 165])
    elif "image_5" in filename:
        lower_bound = np.array([0, 25, 209])
        upper_bound = np.array([162, 173, 255])
    elif "image_6" in filename:
        lower_bound = np.array([25, 18, 44])
        upper_bound = np.array([175, 255, 255])
    elif "image_7" in filename:
        lower_bound = np.array([22, 50, 50])
        upper_bound = np.array([130, 255, 255])
    elif "image_8" in filename:
        lower_bound = np.array([11, 28, 203])
        upper_bound = np.array([19, 54, 255])
    if "image_9" in filename:
        lower_bound = np.array([59, 16, 15])
        upper_bound = np.array([174, 190, 227])
    elif "image_10" in filename:
        lower_bound = np.array([66, 11, 0])
        upper_bound = np.array([179, 221, 217])
    
    return lower_bound, upper_bound

# Process images from the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0)
        
        # Run the image through the Mask R-CNN model
        with torch.no_grad():
            prediction = model(image_tensor)[0]
        
        # Create a binary garment mask
        binary_mask = create_binary_garment_mask(image, prediction)
        
        # Convert PIL image to numpy array for further processing
        image_np = np.array(image)
        
        # Apply custom HSV bounds to filter the garment area
        lower_bound, upper_bound = get_hsv_bounds(filename)
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        garment_hsv_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
        # Combine the HSV mask with the binary garment mask (AND operation)
        final_mask = cv2.bitwise_and(binary_mask, garment_hsv_mask)
        
        # Save the final mask image
        final_mask_image = Image.fromarray(final_mask)
        final_mask_image.save(os.path.join(output_folder, f"mask_{filename}"))

        # Optionally, save the original image with the mask applied
        result_image = cv2.bitwise_and(image_np, image_np, mask=final_mask)
        result_image = Image.fromarray(result_image)
        result_image.save(os.path.join(output_folder, f"processed_{filename}"))

print("Processing complete. Masks and processed images saved to the output folder.")
