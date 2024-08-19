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

## HSV Bounds Calculation

To fine-tune the HSV bounds for each image, I created a Python script that uses the OpenCV GUI to dynamically adjust the HSV ranges. The script allows you to visually inspect the effect of HSV bounds on the image and save the optimal values. 

To use the HSV bounds calculation tool:

1. Place your images in the `input_images/` folder.
2. Run the HSV bounds script:

    ```bash
    python hsv_bounds_calculator.py
    ```

3. Adjust the trackbars to set the lower and upper HSV bounds. Press `s` to save the values for the current image, or `ESC` to move to the next image.

### Example

Here is how the boundary calculator works

![Screenshot 2024-08-17 164438](https://github.com/user-attachments/assets/5825761a-24d9-4496-9697-7ab919b0e128)


## Usage

1. Place the input images in the `input_images/` folder. Supported formats are `.jpg` and `.png`.
   
2. Input custom HSV values for your images that you calculated using the HSV boundary calculator mentioned above.

3. Run the pipeline:

    ```bash
    python garment_masking_pipeline.py
    ``` 

4. The output masks and processed images will be saved in the `output_mask_image/` folder.


## Output

The pipeline produces two types of outputs for each image:

1. **Binary Mask**: A binary image where the garment area is white and the background is black.
2. **Processed Image**: The original image with the binary mask applied, showing only the garment.

### Example

Here are some sample outputs:

- **Input Image**:

  ![1724056628393](https://github.com/user-attachments/assets/fb2d3d89-86fc-4a36-8dd1-225cfe25bafb)


- **Garment Detection**:

![1724056628335](https://github.com/user-attachments/assets/df48869d-b0eb-438c-a680-73cd63a27fb6)


- **Binary Mask**:

![1724056628367](https://github.com/user-attachments/assets/cd9ad92f-3912-495c-b574-e4cec4341c77)

## Model Considerations

- **TRYONline Masking Pipeline** : My project TRYONline- A virtual garment trial room(https://github.com/Jaxkirat/TRYONline), uses a framework called Dressing-in-order (https://github.com/cuiaiyu/dressing-in-order). Since, it was already a built framework, with a number scripts and models inter-linkled with one another. Hence, isolating just the garment-masking-model from the framework was not possible in such a short-time frame.

- **U.Net Model** : I tried building a garment-masking-model using U.net but a robust and accurate model requires a well-labeled and annotated data set. I tried finding a dataset, but the U.net model I was building required the following 3 types of files for each image :
    - The input image
    - A json file that has the garments annotated 
    - And a final masked image
Since, I only had 3 days to produce a binary-garment-masking pipeline, building a robust data-set like I mentioned above was not possible.
Without a data-set the model was very inaccurate and not usable. 
The following are the outputs from the untrained U.net model:

![InShot_20240819_142848439](https://github.com/user-attachments/assets/05256e64-36a8-44ff-8a19-15b2a78edc47)


## Outcomes and Decisions

- **OpenCV**: Since, the above mentioned models were not delivering to my standards, I started working with OpenCV's inbuilt libraries and started differentiating the garments from the background by using the colors and hues from the garment.
  
- **Custom HSV Bounds**: Hence i built a GUI using OpenCV to calculate the HSV value of the sample images and saved these values.

- **Mask-RCNN Threshholding**: While working with the above idea, I was encountering some problems in isolating the garments in some images since the garments and the backgrounds shared the same color. Hence, I used Mask-RCNN for object detection first and then apply HSV bounds to fine-tune the detected garments. A score threshold of 0.5 was used to filter out low-confidence predictions from Mask R-CNN.

## Code Structure

- `garment_masking_pipeline.py`: The main script that processes the images and generates the masks.
- `hsv_bounds_calculator.py`: The script to calculate HSV bounds using the OpenCV GUI.
- `input_images/`: Folder containing the input images.
- `output_mask_image/`: Folder where the output masks and processed images are saved.
- `requirements.txt`: List of required Python packages.

## Future Improvements

- **Model Selection**: With more time and resources, there is a guarantee from me, that I Can build a better model which is much better than the model made by me right now.

- **Data-Set Training** : A robust and well-annotated data set for training, testing and validation can increase the accuracy of this model exponentially.
  
## Conclusion

This pipeline effectively combines the strengths of Mask R-CNN and HSV-based segmentation to accurately mask garment areas in images. It is efficient, easy to use, and could produce high-quality results.

## Author 
- Jaskirat Singh 
