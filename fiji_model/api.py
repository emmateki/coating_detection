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

def find_mask_in_col(x1, maskHeight, mask):
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

    return min_y, max_y

def create_line_roi(maskWidth, maskHeight, output_roi_dir, image_filename, mask, num_rois):
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
            x1=x1,
            y1=y1,
            x2=x1,
            y2=y2,
            float_stroke_width=5.0,
        )
        # Define the path for saving the ROI file
        roi_path = Path(output_roi_dir) / f"{i + 1}.roi"
        roi.tofile(roi_path)
        # Store the ROI file path for later removal
        roi_files.append(roi_path)

    y1 = 200
    y2 = 150
    x = 0

    for i in range(10, 20):  # Generate 10 lines per row
        x1 = space_between_lines / 2 + (x * space_between_lines)
        x += 1
        # Create a line ROI object
        roi = roifile.ImagejRoi(
            roitype=roifile.ROI_TYPE.LINE,
            name=str(i + 1),  # Adjust name to ensure unique names across all rows
            x1=x1,
            y1=y1,
            x2=x1,
            y2=y2,
            float_stroke_width=5.0,
        )
        # Define the path for saving the ROI file
        roi_path = Path(output_roi_dir) / f"{i + 1}.roi"
        roi.tofile(roi_path)
        # Store the ROI file path for later removal
        roi_files.append(roi_path)

    # Create a zip file containing the ROI files
    roi_path_final = Path(output_roi_dir) / f"{image_filename}_line_rois.zip"
    with zipfile.ZipFile(roi_path_final, 'w') as zipf:
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
    roi_path_final = create_line_roi(maskWidth, maskHeight, output_roi_dir, image_filename, mask, num_rois)
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
    lh, uh = int((new_h - h) / 2), int((new_h - h) - (new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int((new_w - w) - (new_w - w) / 2)
    pads = (lw, uw, lh, uh)

    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2]: -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0]: -pad[1]]
    return x

def generate_mask(input_image_path):
    image = cv2.imread(input_image_path)
    parent_dir = os.path.dirname(input_image_path)

    # Crop the image
    image = image[:-120, :]

    test_img = read_image(image)

    model = unet.UNet(depth=5, in_channels=1, start_filters=16)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "trained_model_5_256_16_rev.pth")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = predict(test_img, model, 'cpu')
    
    binary_mask = output.squeeze().numpy()  # Remove batch dimension and convert to numpy array
    return binary_mask

def main():
    parser = argparse.ArgumentParser(description='Take input as path to image, return output in form of path to generated file')
    parser.add_argument('input_image', type=str, help='path to image')
    parser.add_argument('num_rois', type=int, help='number of ROIs')
    args = parser.parse_args()
    input_image_path = args.input_image
    num_rois = args.num_rois

    mask = generate_mask(input_image_path)
    roi_path = roi_maker(mask, input_image_path, num_rois)
    print(roi_path)

if __name__ == "__main__":
    main()
