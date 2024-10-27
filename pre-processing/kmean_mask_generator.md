---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Corrosion
    language: python
    name: python3
---

## Preparation



### 0.Lib


```python
import os
import zipfile
import shutil
from PIL import Image
from read_roi import read_roi_file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import cv2
import numpy as np
from sklearn.cluster import KMeans
```

### 1.Paths

```python
imagesFolder = ""
maskFolder = ""
roiFolder = ""
```

### 2. tif -> png

```python
def convertPng(tif_folder):
    tif_files = [f for f in os.listdir(tif_folder) if f.endswith(".tif")]
    for tifFile in tif_files:
        tifPath = os.path.join(tif_folder, tifFile)

        pngFileName = os.path.splitext(tifFile)[0] + ".png"
        pngPath = os.path.join(tif_folder, pngFileName)

        img = Image.open(tifPath)
        img.save(pngPath, "PNG")

        os.remove(tifPath)


convertPng(imagesFolder)
```

### 3. Crop 120 pixels

```python
def crop_images(imagesFolder):
    for filename in os.listdir(imagesFolder):
        if filename.endswith(".png"):
            image_path = os.path.join(imagesFolder, filename)
            image = cv2.imread(image_path)

            height, _, _ = image.shape
            cropped_image = image[: height - 120, :, :]

            cropped_image_path = os.path.join(imagesFolder, filename)
            cv2.imwrite(cropped_image_path, cropped_image)


crop_images(imagesFolder)
```

### 4. Show k means

showing color maps to better customize the kmeans algorithm


```python
def k_mean_mask(imagesFolder, num_clusters=3):
    for filename in os.listdir(imagesFolder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(imagesFolder, filename)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
            gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

            # Ensure correct data type
            reshaped_image = np.ascontiguousarray(
                gray_image.reshape((-1, 1)).astype(np.float32)
            )

            # KMeans clustering
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            kmeans.fit(reshaped_image)

            labels = kmeans.labels_
            labels = labels.reshape(gray_image.shape)

            # Generate colors for clusters
            colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)

            # Create a blank image for clustered result
            clustered_image = np.zeros_like(image, dtype=np.uint8)

            # Assign cluster colors
            for i in range(num_clusters):
                clustered_image[labels == i] = colors[i]

            # Debugging print statements
            print(f"clustered_image type: {type(clustered_image)}")
            print(f"clustered_image shape: {clustered_image.shape}")
            print(f"clustered_image dtype: {clustered_image.dtype}")

            # Ensure it's a valid NumPy array before converting colors
            if isinstance(clustered_image, np.ndarray):
                # Convert BGR to RGB for plotting
                plt.imshow(cv2.cvtColor(clustered_image, cv2.COLOR_BGR2RGB))
                plt.show()
            else:
                print("clustered_image is not a valid NumPy array.")


# Call the function with your images folder
k_mean_mask(imagesFolder)
```

### 5. Mask Generator

```python
num_clusters = 3


def k_mean_mask(imagesFolder, maskFolder, num_clusters):

    if not os.path.exists(maskFolder):
        os.makedirs(maskFolder)

    for filename in os.listdir(imagesFolder):
        if filename.endswith(".png"):
            image_path = os.path.join(imagesFolder, filename)

            image = cv2.imread(image_path)

            blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
            gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
            reshaped_image = np.ascontiguousarray(
                gray_image.reshape((-1, 1)).astype(np.float32)
            )

            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            kmeans.fit(reshaped_image)

            labels = kmeans.labels_
            unique_labels, label_counts = np.unique(labels, return_counts=True)

            # to determine which label belong to coating
            centroids = kmeans.cluster_centers_
            distances_to_upper_boundary = [centroid[0] for centroid in centroids]
            closest_to_upper_boundary = np.argmin(distances_to_upper_boundary)

            label_counts_sorted_indices = np.argsort(label_counts)
            largest_cluster_label = unique_labels[label_counts_sorted_indices[-1]]

            # Remove the closest to upper boundary, the biggest, and the second closest to upper boundary
            remaining_labels = [
                label
                for label in unique_labels
                if label not in [closest_to_upper_boundary, largest_cluster_label]
            ]

            # Color the remaining two clusters to white
            cluster_colors = {label: [255, 255, 255] for label in remaining_labels}

            segmented_img = np.zeros_like(image)

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    label = labels[i * image.shape[1] + j]
                    if label in cluster_colors:
                        segmented_img[i, j] = cluster_colors[label]

            #  polishing directly on the segmented image
            mask = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                thresh, connectivity=8
            )
            touching_both_sides = False
            for i in range(1, num_labels):
                left_side = labels[:, 0] == i
                right_side = labels[:, -1] == i
                if np.any(left_side) and np.any(right_side):
                    touching_both_sides = True
                    break
            if touching_both_sides:
                largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                largest_component_mask = (
                    np.uint8(labels == largest_component_index) * 255
                )
                filtered_mask = cv2.bitwise_and(thresh, largest_component_mask)
            else:
                filtered_mask = mask
            filtered_mask = cv2.cvtColor(filtered_mask, cv2.IMREAD_GRAYSCALE)
            # Convert the filtered mask to binary
            _, binary_mask = cv2.threshold(filtered_mask, 127, 255, cv2.THRESH_BINARY)
            binary_mask_path = os.path.join(maskFolder, f"{filename}")
            cv2.imwrite(binary_mask_path, binary_mask)


k_mean_mask(imagesFolder, maskFolder, num_clusters)
```

### 5.Show

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

imagesFolder = "/home/emma/Documents/DATA/kmean/train/train_x"
maskFolder = "/home/emma/Documents/DATA/kmean/train/train_y"  # Removed extra space


def show(image_folder, mask_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            original_image = cv2.imread(image_path)

            if original_image is None:
                print(f"Error loading image: {image_path}")
                continue

            mask_path = os.path.join(mask_folder, filename)
            binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if binary_mask is None:
                print(f"Error loading mask: {mask_path}")
                continue

            # Convert the binary mask to 3 channels
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

            # Apply red color to the binary mask
            binary_mask[np.where((binary_mask == [255, 255, 255]).all(axis=2))] = [
                0,
                0,
                255,
            ]
            # Red mask half-transparent
            alpha = 0.5
            overlay = cv2.addWeighted(binary_mask, alpha, original_image, 1 - alpha, 0)

            plt.figure()
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"Sample with mask: {filename}")
            plt.show()

            # If save uncomment
            # image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            # mask_Path = os.path.join('path' + filename)
            # cv2.imwrite(mask_Path, image)


show(imagesFolder, maskFolder)
```
