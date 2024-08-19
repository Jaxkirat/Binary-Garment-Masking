# Garment Masking Pipeline

This repository contains a garment masking pipeline designed to accurately extract garment areas from a set of images. The pipeline produces binary masks for each image, highlighting the garment in white and the background in black.

## Overview

The pipeline utilizes a combination of Mask R-CNN and custom HSV (Hue, Saturation, Value) bounds to create accurate binary masks for garments in input images. The masks are then refined by applying specific HSV bounds tailored to each image. The final output consists of binary masks and optionally processed images with the masks applied.

## Requirements

- Python 3.7 or higher
- Required Python packages (install via `requirements.txt`):
  - OpenCV
  - Torch
  - Torchvision
  - Pillow
  - NumPy

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/garment-masking-pipeline.git
    cd garment-masking-pipeline
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place the input images in the `input_images/` folder. Supported formats are `.jpg` and `.png`.

2. Run the pipeline:

    ```bash
    python garment_masking_pipeline.py
    ```

3. The output masks and processed images will be saved in the `output_mask_image/` folder.

## HSV Bounds Calculation

To fine-tune the HSV bounds for each image, I created a Python script that uses the OpenCV GUI to dynamically adjust the HSV ranges. The script allows you to visually inspect the effect of HSV bounds on the image and save the optimal values. Screenshots of this process will be added to the repository.

To use the HSV bounds calculation tool:

1. Place your images in the `input_images/` folder.
2. Run the HSV bounds script:

    ```bash
    python hsv_bounds_calculator.py
    ```

3. Adjust the trackbars to set the lower and upper HSV bounds. Press `s` to save the values for the current image, or `ESC` to move to the next image.

###Example

Here is how the boundary calculator works

![Screenshot 2024-08-17 164438](https://github.com/user-attachments/assets/5825761a-24d9-4496-9697-7ab919b0e128)


## Output

The pipeline produces two types of outputs for each image:

1. **Binary Mask**: A binary image where the garment area is white and the background is black.
2. **Processed Image**: The original image with the binary mask applied, showing only the garment.

### Example

Here are some sample outputs:

- **Input Image**: image_1.jpg
- **Binary Mask**: mask_image_1.jpg
- **Processed Image**: processed_image_1.jpg

## Assumptions and Decisions

- **Pretrained Model**: The Mask R-CNN model is pretrained on COCO, which provides a good starting point for detecting garments.
- **Custom HSV Bounds**: Custom HSV bounds were defined based on the filename to refine the mask for each image.
- **Thresholding**: A score threshold of 0.5 was used to filter out low-confidence predictions from Mask R-CNN.

## Code Structure

- `garment_masking_pipeline.py`: The main script that processes the images and generates the masks.
- `hsv_bounds_calculator.py`: The script to calculate HSV bounds using the OpenCV GUI.
- `input_images/`: Folder containing the input images.
- `output_mask_image/`: Folder where the output masks and processed images are saved.
- `requirements.txt`: List of required Python packages.

## Future Improvements

- **Dynamic HSV Bound Adjustment**: Implement a more dynamic approach to adjust HSV bounds based on the image content.
- **Model Fine-Tuning**: Fine-tune the Mask R-CNN model on a dataset specific to garment detection for improved accuracy.

## Conclusion

This pipeline effectively combines the strengths of Mask R-CNN and HSV-based segmentation to accurately mask garment areas in images. It is efficient, easy to use, and produces high-quality results.

## Contact

For any questions or issues, please contact [your email].
