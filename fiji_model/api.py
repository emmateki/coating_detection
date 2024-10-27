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
import unet
import glob
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import label, regionprops

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
    y1_default=250,
    y2_default=200,
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

    # Generate and save individual line ROIs
    for i in range(num_rois):
        x1 = round(space_between_lines / 2 + (i * space_between_lines))
        y1, y2 = find_mask_in_col(x1, maskHeight, mask)
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

    y1 = y1_default
    y2 = y2_default
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
    # Convert to binary mask with this threshold
    res_unp_binary = (res_unp > 0.8).float()
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

    out = F.pad(x, pads, "constant", 0)

    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2] : -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0] : -pad[1]]
    return x


def get_clusters(binary_mask):
    """
    Label the clusters in the binary mask and return labeled mask and number of clusters.

    Args:
        binary_mask (numpy.ndarray): Binary mask to be labeled.

    Returns:
        tuple: Labeled mask and the number of clusters.
    """
    labeled_mask = label(binary_mask, connectivity=2)
    num_clusters = labeled_mask.max()
    return labeled_mask, num_clusters


def find_cluster_touches_sides(labeled_mask):
    """
    Find if any cluster touches both left and right sides of the mask.

    Args:
        labeled_mask (numpy.ndarray): Mask with labeled clusters.

    Returns:
        numpy.ndarray or None: The cluster touching both sides, or None if no such cluster exists.
    """
    regions = regionprops(labeled_mask)
    for region in regions:
        min_col, min_row, max_col, max_row = region.bbox
        if min_col == 0 and max_col == labeled_mask.shape[1]:
            return labeled_mask == region.label
    return None


def find_largest_clusters(labeled_mask, side):
    """
    Find the largest clusters that touch a specific side ('left' or 'right').

    Args:
        labeled_mask (numpy.ndarray): Mask with labeled clusters.
        side (str): The side to check ('left' or 'right').

    Returns:
        numpy.ndarray or None: The largest cluster touching the specified side, or None if no such cluster exists.
    """
    regions = regionprops(labeled_mask)
    if side == "left":
        side_col = 0
    elif side == "right":
        side_col = labeled_mask.shape[1] - 1
    else:
        raise ValueError("Side must be either 'left' or 'right'")

    side_clusters = [
        region
        for region in regions
        if region.bbox[1] <= side_col <= region.bbox[3]
        or region.bbox[1] <= side_col <= region.bbox[3]
    ]

    if side_clusters:
        largest_cluster = max(side_clusters, key=lambda r: r.area)
        return labeled_mask == largest_cluster.label

    return None


def generate_mask(
    input_image_path,
    depth=4,
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
        crop_down_percentage (int): Percentage of height that consists of the title and details of the image.

    Returns:
        numpy.ndarray: Binary mask.
    """
    image = cv2.imread(input_image_path)
    height, width = image.shape[:2]

    image = image[: round(-crop_down_percentage * width), :]

    test_img = read_image(image)

    decoder_channels = [start_filters * 2**i for i in range(depth, 0, -1)]
    model = Unet(
        encoder_depth=depth,
        encoder_weights="imagenet",
        in_channels=1,
        decoder_channels=decoder_channels,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pth_files = glob.glob(os.path.join(script_dir, "*.pth"))
    model_path = pth_files[0]

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = predict(test_img, model, "cpu")

    # Remove batch dimension and convert to numpy array
    binary_mask = output.squeeze().numpy()

    cv2.imwrite("binary_mask.png", (binary_mask * 255).astype(np.uint8))

    labeled_mask, num_clusters = get_clusters(binary_mask)

    imageio.imwrite("binary_mask.png", (binary_mask * 255).astype(np.uint8))

    final_mask = find_cluster_touches_sides(labeled_mask)
    if final_mask is not None:
        return final_mask

    left_mask = find_largest_clusters(labeled_mask, "left")
    right_mask = find_largest_clusters(labeled_mask, "right")

    if left_mask is not None:
        final_mask = np.logical_or(final_mask, left_mask)

    if right_mask is not None:
        final_mask = np.logical_or(final_mask, right_mask)

    if final_mask is None:
        largest_cluster = max(regionprops(labeled_mask), key=lambda r: r.area)
        final_mask = labeled_mask == largest_cluster.label

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
