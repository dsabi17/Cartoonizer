
# Image Processing Pipeline DRAFT

This project implements a custom image processing pipeline in Python, designed to perform various image transformations, including median cut color quantization, blurring, grayscale conversion, edge detection, and hysteresis thresholding.

## Features

- **Median Cut Color Quantization**: Reduces the number of colors in an image while maintaining visual quality.
- **Gaussian Blur**: Smoothens the image to reduce noise and detail.
- **Grayscale Conversion**: Converts images to grayscale for edge detection.
- **Sobel Edge Detection**: Identifies edges in the image by calculating intensity gradients.
- **Non-Maxima Suppression**: Refines edges by keeping only the most significant ones.
- **Hysteresis Thresholding**: Enhances edge detection by linking weak edges to strong ones.
- **Outline Addition**: Combines the processed borders with the original image to create a visually enhanced result.

The script will process each image in the `photos` folder and save the results in separate directories for each image. Each directory will contain the following:

   - Original image.
   - Median cut quantized image.
   - Blurred images (median cut and original).
   - Grayscale version of the blurred image.
   - Sobel edge-detected image.
   - Non-maxima suppressed image.
   - Hysteresis thresholded image.
   - Final combined image with borders.

## Project Structure

- **display_image**: Displays the image using Matplotlib.
- **grayscale**: Converts an image to grayscale.
- **blur**: Applies Gaussian blur to an image.
- **sobel**: Performs Sobel edge detection.
- **non_maxima**: Refines edges using non-maxima suppression.
- **hysteresis_thresholding**: Enhances edge detection using thresholding.
- **median_cut**: Reduces the number of colors in an image.
- **add_outline_to_image**: Combines borders with the original image.

