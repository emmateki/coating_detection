import os
import numpy as np
import argparse
from pathlib import Path
import roifile
import zipfile
import cv2
from sklearn.cluster import KMeans


def find_mask_in_col(x1, maskHeight, mask_path):
    min = maskHeight
    max = 0
    # Load the binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    for y in range (0, maskHeight):
        # get number what is saved in mask balck or white 
        is_mask = mask [y, x1]
        
        if is_mask != 0:
            if (y <min):
                min = y
            elif (y>max):
                max = y
        y= y+1

    return min, max ;

def create_line_roi (maskWidth, maskHeight, output_roi_dir, image_filename,mask_path,num_rois):
    """
    Create lines (ROIs) and store them in a zip file. 

    This function generates a specified number of line ROIs on an image 
    of given dimensions and saves them as individual ROI files. It then
    creates a zip archive containing these ROI files and removes the
    individual ROI files from the filesystem.

    Args:
        imageWidth (int): Width of the image canvas.
        imageHeight (int): Height of the image canvas.
        output_dir (str): Directory where the ROI files and the zip archive will be stored.
        
    Returns:
        None
    """
    # Calculate the space between lines based on image width
    space_between_lines = maskWidth / num_rois
    # Array for saving the roi file paths
    roi_files = []  

    # Generate and save individual line ROIs
    for i in range(num_rois):
        x1 = round (space_between_lines / 2 + (i * space_between_lines))
        y1, y2 = find_mask_in_col(x1, maskHeight, mask_path);
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

    for i in range(10,20):  # Generate 10 lines per row
        #if i % 10 == 0 and i != 0:  # Move to the next row every 10 lines (excluding the first iteration)
           # y1 += 200
           # y2 += 200
           # x = 0  # Reset x-coordinate for each new row
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


def roi_maker (input_image_path, num_rois,num_clusters):
    mask_path = generate_mask(input_image_path,num_clusters)

    mask = cv2.imread(mask_path)
    maskHeight, maskWidth, _ = mask.shape

    image_dir, image_filename = os.path.split(input_image_path)
    image_filename = Path (image_filename).stem
    output_roi_dir = image_dir
    roi_path_final = create_line_roi (maskWidth, maskHeight, output_roi_dir, image_filename,mask_path,num_rois)
    #remove mask
    os.remove(mask_path)
    return roi_path_final


def generate_mask(input_image_path,num_clusters):
    # Load the input image
    image = cv2.imread(input_image_path)
    parent_dir = os.path.dirname(input_image_path)

    # Crop the image - the description down
    image = image[:-120, :]

    # Perform KMeans clustering
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    reshaped_image = gray_image.reshape((-1, 1))
    
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(reshaped_image)
    labels = kmeans.labels_
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    # to determine which label belong to coating
    centroids = kmeans.cluster_centers_
    distances_to_upper_boundary = [centroid[0] for centroid in centroids]
    closest_to_upper_boundary = np.argmin(distances_to_upper_boundary)

    label_counts_sorted_indices = np.argsort(label_counts)

    smallest_cluster_label = unique_labels[label_counts_sorted_indices[0]]
    middle_cluster_label = unique_labels[label_counts_sorted_indices[1]]
    largest_cluster_label = unique_labels[label_counts_sorted_indices[2]]

    cluster_colors = {smallest_cluster_label: [0, 0, 0], largest_cluster_label: [0, 0, 0], middle_cluster_label: [0, 0, 0]}

    if closest_to_upper_boundary == middle_cluster_label:
        cluster_colors[smallest_cluster_label] = [255, 255, 255]
    else:
        cluster_colors[middle_cluster_label] = [255, 255, 255]

    segmented_img = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label = labels[i * image.shape[1] + j]
            segmented_img[i, j] = cluster_colors[label]

    #  polishing directly on the segmented image
    mask = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    touching_both_sides = False
    for i in range(1, num_labels):
        left_side = labels[:, 0] == i
        right_side = labels[:, -1] == i
        if np.any(left_side) and np.any(right_side):
            touching_both_sides = True
            break
    if touching_both_sides:
        largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_component_mask = np.uint8(labels == largest_component_index) * 255
        filtered_mask = cv2.bitwise_and(thresh, largest_component_mask)
    else:
        filtered_mask = mask
    filtered_mask = cv2.cvtColor(filtered_mask, cv2.IMREAD_GRAYSCALE)
    # Convert the filtered mask to binary
    _, binary_mask = cv2.threshold(filtered_mask, 127, 255, cv2.THRESH_BINARY)

    binary_mask_path = os.path.join(parent_dir, "binary_mask_temp.png")
    cv2.imwrite(binary_mask_path, binary_mask)

    return binary_mask_path
    


def main():
    parser = argparse.ArgumentParser(description='Take input as path to image, return output in form of path to generated file')
    parser.add_argument('input_image', type=str, help = 'path to image')
    parser.add_argument('num_rois', type=int, help='number of ROIs')
    args = parser.parse_args()
    input_image_path = args.input_image
    num_rois = args.num_rois

    # can be add as argument
    num_clusters = 3

    roi_path = roi_maker(input_image_path,num_rois,num_clusters)
    print (roi_path)


if __name__ == "__main__":
    main()