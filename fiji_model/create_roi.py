from PIL import Image
import os
import roifile
import zipfile
from pathlib import Path
import argparse


def create_line_roi(imageWidth, imageHeight, output_dir, image_path,  num_lines=10, num_lines_per_row=10, y1_default=100, y2_default=50, float_stroke_width=5.0, space_between_rows=200):
    """
    Create lines (ROIs) and store them in a zip file. 

    This function generates a specified number of line ROIs on an image 
    of given dimensions and saves them as individual ROI files. It then
    creates a zip archive containing these ROI files and removes the
    individual ROI files from the filesystem.

    Args:
        imageWidth (int): Width of the image canvas.
        imageHeight (int): Height of the image canvas.
        num_lines (int): Number of lines ROIs to create.
        output_dir (str): Directory where the ROI files and the zip archive will be stored.
        num_lines_per_row (int): Number of lines per row. Default is 10.
        y1_default (int): Default starting y-coordinate for the first set of ROIs. 
        y2_default (int): Default ending y-coordinate for the first set of ROIs. 
        float_stroke_width (float): Stroke width for the lines. Default is 5.0.
        space_between_rows (int): Space between rows. Default is 200.

    Returns:
        None
    """
    # Calculate the space between lines based on image width
    space_between_lines = imageWidth / num_lines_per_row
    # Array for saving the roi file paths
    roi_files = []

    no_of_line = 0

    for i in range(num_lines):  # Generate 10 lines per row
        # Move to the next row every 10 lines (excluding the first iteration)
        if i % num_lines_per_row == 0 and i != 0:
            y1_default += space_between_rows
            y2_default += space_between_rows
            x = 0  # Reset x-coordinate for each new row
        x1 = int(space_between_lines / 2 + (x * space_between_lines))
        no_of_line += 1
        # Create a line ROI object
        roi = roifile.ImagejRoi(
            roitype=roifile.ROI_TYPE.LINE,
            # Adjust name to ensure unique names across all rows
            name=str(i + 1),
            x1=x1,
            y1=y1_default,
            x2=x1,
            y2=y2_default,
            float_stroke_width=float_stroke_width,
        )
        # Define the path for saving the ROI file
        roi_path = Path(output_dir) / f"{i + 1}.roi"
        roi.tofile(roi_path)
        # Store the ROI file path for later removal
        roi_files.append(roi_path)

    # Create a zip file containing the ROI files
    image_name = Path(image_path).stem
    zip_filename = Path(output_dir) / f"{image_name}_line_rois.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for roi_path in roi_files:
            # Write each ROI file to the zip archive
            zipf.write(roi_path, roi_path.name)
    # Remove the individual ROI files
    for roi_path in roi_files:
        roi_path.unlink()


def main(image_path):
    """
    This function reads an image from the specified path, calculates its dimensions. 
    Calls the def function create_line_roi.
    The number of line ROIs and the output directory are predefined here.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        None
    """
    # Extract directory and filename from the input image path
    image_dir, image_filename = os.path.split(image_path)
    output_dir = image_dir
    # Open the image and get its dimensions
    image = Image.open(image_path)
    imageWidth, imageHeight = image.size

    # Generate line ROIs on the image and save as a zip archive
    create_line_roi(imageWidth, imageHeight, output_dir, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create line ROIs on an image and save as a zip archive")
    parser.add_argument("image_path", type=str,
                        help="Path to the input image file")
    args = parser.parse_args()
    main(args.image_path)
