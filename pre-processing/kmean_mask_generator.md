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
imagesFolder = "/k_mean/images"
maskFolder = "k_mean/output"
roiFolder = "data/ROI"
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

+- the title and legend in the botton of the images.

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

Showing color maps to better customize the kmeans algorithm.


```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def k_mean_mask(imagesFolder, num_clusters=3):
    for filename in os.listdir(imagesFolder):
        if filename.endswith(".png"):
            image_path = os.path.join(imagesFolder, filename)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
            gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
            reshaped_image = gray_image.reshape((-1, 1)).astype(np.float32)
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            kmeans.fit(reshaped_image)

            labels = kmeans.labels_.reshape(gray_image.shape)
            colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)

            # Create a blank image for the clustered result
            clustered_image = np.zeros_like(image, dtype=np.uint8)

            for i in range(num_clusters):
                clustered_image[labels == i] = colors[i]

            plt.imshow(cv2.cvtColor(clustered_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Clustered Image: {filename}")
            plt.axis("off")
            plt.show()


k_mean_mask(imagesFolder)
```

### 5. Mask Generator

```python
num_clusters = 3


def k_mean_mask(imagesFolder, maskFolder, num_clusters=3):
    if not os.path.exists(maskFolder):
        os.makedirs(maskFolder)

    for filename in os.listdir(imagesFolder):
        if filename.endswith(".png"):
            image_path = os.path.join(imagesFolder, filename)

            # Read, blur, and convert to grayscale
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(
                cv2.GaussianBlur(image, (7, 7), 0), cv2.COLOR_BGR2GRAY
            )
            reshaped_image = gray_image.reshape(-1, 1).astype(np.float32)

            kmeans = KMeans(n_clusters=num_clusters, n_init=10).fit(reshaped_image)
            labels = kmeans.labels_.reshape(gray_image.shape)

            # Determine the cluster to keep - This needs to be customied based on the type of coating - as well as num_clusters

            centroids = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centroids)
            middle_index = sorted_indices[num_clusters // 2]

            # Create a mask for the middle cluster - where the coating is
            mask = np.where(labels == middle_index, 255, 0).astype(np.uint8)

            # Check if any region touches both sides - to keep just teh biggest one without any small part
            num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            touching_both_sides = any(
                np.any(label_img[:, 0] == i) and np.any(label_img[:, -1] == i)
                for i in range(1, num_labels)
            )

            if touching_both_sides:
                largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask = np.where(label_img == largest_component_index, 255, 0).astype(
                    np.uint8
                )

            cv2.imwrite(os.path.join(maskFolder, filename), mask)


k_mean_mask(imagesFolder, maskFolder, num_clusters)
```

### 5.Show

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

imagesFolder = "/home/emma/Documents/DATA/kmean/train/train_x"
maskFolder = "/home/emma/Documents/DATA/kmean/train/train_y"


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
