import os
import numpy as np
import argparse
from pathlib import Path
import roifile
import zipfile
import cv2
import torch
from torch.functional import F
import imageio
import glob
import matplotlib.pyplot as plt

import segmentation_models_pytorch
from segmentation_models_pytorch import Unet


def find_mask_in_col(
    x1,
    maskHeight,
    mask,
    y1_default=150,
    y2_default=100,
):
    """
    Find the minimum and maximum y-coordinates in a column of the mask.

    Args:
        x1 (int): The x-coordinate of the column.
        maskHeight (int): The height of the mask.
        mask (numpy.ndarray): The binary mask.
        y1_default (int): Default starting y-coordinate for the first set of ROIs.
        y2_default (int): Default ending y-coordinate for the first set of ROIs.


    Returns:
        tuple: A tuple containing the minimum and maximum y-coordinates.
    """
    min_y = maskHeight
    max_y = 0

    for y in range(maskHeight):
        # Get the pixel value in the mask (black or white)
        is_mask = mask[y, x1]

        if is_mask != 0:
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
    # if no mask was found get default
    if min_y == maskHeight and max_y == 0:
        min_y = y1_default
        max_y = y2_default

    return min_y, max_y


def create_line_roi(
    maskWidth,
    maskHeight,
    output_roi_dir,
    image_filename,
    mask,
    num_rois,
    float_stroke_width=5.0,
    start_of_line=0.5,
):
    """
    Create lines (ROIs) and store them in a zip file.

    This function generates a specified number of line ROIs on an image
    of given dimensions and saves them as individual ROI files. It then
    creates a zip archive containing these ROI files and removes the
    individual ROI files from the filesystem.

    Args:
        maskWidth (int): Width of the mask.
        maskHeight (int): Height of the mask.
        output_roi_dir (str): Directory where the ROI files and the zip archive will be stored.
        image_filename (str): Filename of the input image.
        mask (numpy.ndarray): Binary mask.
        num_rois (int): Number of ROIs to generate.
        y1_default (int): Default starting y-coordinate for the second set of ROIs. Default is 250. This set is not predicted but added for manual oxidation layer.
        y2_default (int): Default ending y-coordinate for the second set of ROIs. Default is 200.
        float_stroke_width (float): Stroke width for the lines. Default is 5.0.
        start_of_line (float): # the line needs to start at the edge of one pixel not in the middle


    Returns:
        Path: Path to the zip file containing the ROIs.
    """
    # Calculate the space between lines based on the mask width
    space_between_lines = maskWidth / num_rois
    # Array for saving the ROI file paths
    roi_files = []
    y2_mean = []

    # Generate and save individual line ROIs
    for i in range(num_rois):
        x1 = round(space_between_lines / 2 + (i * space_between_lines))
        y1, y2 = find_mask_in_col(x1, maskHeight, mask)
        y2_mean.append(y2)
        # Create a line ROI object
        roi = roifile.ImagejRoi(
            roitype=roifile.ROI_TYPE.LINE,
            name=str(i + 1),
            x1=int(x1),
            y1=int(y1) - start_of_line,
            x2=int(x1),
            y2=int(y2) + start_of_line,
            float_stroke_width=float_stroke_width,
        )
        # Define the path for saving the ROI file
        roi_path = Path(output_roi_dir) / f"{i + 1}.roi"
        roi.tofile(roi_path)
        # Store the ROI file path for later removal
        roi_files.append(roi_path)

    # dynamically set y1 and y2 for the second set
    y2 = np.mean(y2_mean) + 50
    y1 = y2 + 50
    x = 0

    for i in range(num_rois, num_rois * 2):  # Generate 10 lines per row
        x1 = space_between_lines / 2 + (x * space_between_lines)
        x += 1
        # Create a line ROI object
        roi = roifile.ImagejRoi(
            roitype=roifile.ROI_TYPE.LINE,
            # Adjust name to ensure unique names across all rows
            name=str(i + 1),
            x1=x1,
            y1=y1 - start_of_line,
            x2=x1,
            y2=y2 + start_of_line,
            float_stroke_width=float_stroke_width,
        )
        # Define the path for saving the ROI file
        roi_path = Path(output_roi_dir) / f"{i + 1}.roi"
        roi.tofile(roi_path)
        # Store the ROI file path for later removal
        roi_files.append(roi_path)

    # Create a zip file containing the ROI files
    roi_path_final = Path(output_roi_dir) / f"{image_filename}_line_rois.zip"
    with zipfile.ZipFile(roi_path_final, "w") as zipf:
        for roi_path in roi_files:
            # Write each ROI file to the zip archive
            zipf.write(roi_path, roi_path.name)
    # Remove the individual ROI files
    for roi_path in roi_files:
        roi_path.unlink()

    return roi_path_final


def roi_maker(mask, input_image_path, num_rois):
    maskHeight, maskWidth = mask.shape

    image_dir, image_filename = os.path.split(input_image_path)
    image_filename = Path(image_filename).stem
    output_roi_dir = image_dir
    roi_path_final = create_line_roi(
        maskWidth, maskHeight, output_roi_dir, image_filename, mask, num_rois
    )
    return roi_path_final


def imread(img):
    if img.ndim == 3:
        img = img[:, :, 0]

    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = (img - img_min) / (img_max - img_min)

    return np.float32(img_norm)


def read_image(image_path):
    img = imread(image_path)
    half = np.maximum(img.shape[0] // 2, 256)
    img = img[:half]
    return img


def predict(img, model, device, pad_stride=32):
    img_3d = np.stack([img] * 1)
    tensor = torch.from_numpy(img_3d).to(device)[None]
    padded_tensor, pads = pad_to(tensor, pad_stride)
    res_tensor = model(padded_tensor)
    res_unp = unpad(res_tensor, pads)
    # activation function
    res_unp = torch.sigmoid(res_unp)
    res_unp_binary = (res_unp > 0.5).float()
    return res_unp_binary.squeeze(0).squeeze(0)


def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h - h) // 2), int((new_h - h) - (new_h - h) // 2)
    lw, uw = int((new_w - w) // 2), int((new_w - w) - (new_w - w) // 2)
    pads = (lw, uw, lh, uh)

    out = F.pad(x, pads, "replicate", 0)

    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2] : -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0] : -pad[1]]
    return x


def generate_mask(
    input_image_path,
    depth=5,
    in_channels=1,
    start_filters=16,
    crop_down_percentage=0.117,
):
    """
    Generate a binary mask from the input image using a U-Net model.
    Post-processing steps:
    1. Identify and retain the largest cluster that touches both the left and right sides of the mask.
    2. Identify and retain the largest clusters touching the left and/or right side, if not covered in step 1.
    3. If no clusters touching both sides are found, retain the largest cluster from the entire mask.

    Args:
        input_image_path (str): Path to the input image.
        depth (int): Depth of the U-Net model.
        in_channels (int): Number of input channels for the U-Net model.
        start_filters (int): Number of starting filters for the U-Net model.
        crop_down_percentage (float): Percentage of height that consists of the title and details of the image.
        Was calculated by the ratio of the title height to the total height of the image.

    Returns:
        numpy.ndarray: Binary mask.
    """
    image = cv2.imread(input_image_path)
    height, width = image.shape[:2]

    # Crop out unwanted parts from the image based on the given percentage
    image = image[: round(-crop_down_percentage * height), :]
    test_img = read_image(image)

    # Prepare the U-Net model
    decoder_channels = [start_filters * 2**i for i in range(depth, 0, -1)]
    model = Unet(
        encoder_depth=depth,
        encoder_weights="imagenet",
        in_channels=1,
        decoder_channels=decoder_channels,
    )

    # Load the model weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pth_files = glob.glob(os.path.join(script_dir, "*.pth"))
    model_path = pth_files[0]
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Generate prediction
    with torch.no_grad():
        output = predict(test_img, model, "cpu")

    # Convert to binary mask
    binary_mask = output.squeeze().numpy()

    return binary_mask


def main():
    parser = argparse.ArgumentParser(
        description="Take input as path to image, return output in form of path to generated file"
    )
    parser.add_argument("input_image", type=str, help="path to image")
    parser.add_argument("num_rois", type=int, help="number of ROIs per one row")
    args = parser.parse_args()
    input_image_path = args.input_image
    num_rois = args.num_rois

    mask = generate_mask(input_image_path)
    roi_path = roi_maker(mask, input_image_path, num_rois)
    # print so it will be taken as input for FIJI
    print(roi_path)


if __name__ == "__main__":
    main()
