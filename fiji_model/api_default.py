from PIL import Image
import os
import roifile
import zipfile
from pathlib import Path
import argparse


def create_line_roi(imageWidth, imageHeight, num_lines, output_dir, image_path):
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

    Returns:
        None
    """
    # Calculate the space between lines based on image width
    space_between_lines = imageWidth / 10
    roi_files = []

    y1 = 100
    y2 = 50

    x = 0

    for i in range(num_lines):  # Generate 10 lines per row
        if (
            i % 10 == 0 and i != 0
        ):  # Move to the next row every 10 lines (excluding the first iteration)
            y1 += 200
            y2 += 200
            x = 0
        x1 = space_between_lines / 2 + (x * space_between_lines)
        x += 1
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
        roi_path = Path(output_dir) / f"{i + 1}.roi"
        roi.tofile(roi_path)
        roi_files.append(roi_path)

    # Create a zip file containing the ROI files
    image_name = Path(image_path).stem
    zip_filename = Path(output_dir) / f"{image_name}_line_rois.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for roi_path in roi_files:
            # Write each ROI file to the zip archive
            zipf.write(roi_path, roi_path.name)
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
    image_dir, image_filename = os.path.split(image_path)
    output_dir = image_dir
    image = Image.open(image_path)
    imageWidth, imageHeight = image.size
    # Number of line ROIs to generate
    num_lines = 30
    create_line_roi(imageWidth, imageHeight, num_lines, output_dir, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create line ROIs on an image and save as a zip archive"
    )
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    args = parser.parse_args()
    main(args.image_path)
