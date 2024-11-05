---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Polygon mask generator 

This notebook will need unziped roi folder and images. 
They need to have same names but roi folders have *_measurements


```python
import matplotlib.pyplot as plt
from read_roi import read_roi_file
from scipy.ndimage import zoom
import os
import numpy as np
import cv2
from read_roi import read_roi_file
```

```python
roiFolder = ""
outputFolder = ""
imagesFolder = ""

numStartRoi = 0
numEndRoi = 10
```

### 1.Show the ROI lines from ROI folder

This is just for showing - can be commented

```python
def showMeasurements(roiFolder, imageFolder, numStartRoi, numEndRoi):
    for foldername in os.listdir(roiFolder):
        if foldername.endswith("_measurements"):
            measurement_folder_path = os.path.join(roiFolder, foldername)

            # Corresponding image name
            image_name = foldername.replace("_measurements", ".png")
            image_path = os.path.join(imageFolder, image_name)

            if not os.path.exists(image_path):
                print(f"Image {image_name} not found in {imageFolder}.")
                continue

            img = plt.imread(image_path)
            plt.imshow(img)

            roi_files = [
                f for f in os.listdir(measurement_folder_path) if f.endswith(".roi")
            ]

            for roiFile in roi_files:
                # Extract the number from the file name
                number = int(roiFile.split(".")[0])
                if numStartRoi < number <= numEndRoi:
                    roi_path = os.path.join(measurement_folder_path, roiFile)
                    roi = read_roi_file(roi_path)

                    # Plot each ROI
                    for key, value in roi.items():
                        xValues = value["x1"], value["x2"]
                        yValues = value["y1"], value["y2"]
                        plt.plot(xValues, yValues, "r")  # Red lines for ROIs

            plt.title(f"Measurements for {image_name}")
            plt.show()


showMeasurements(roiFolder, imagesFolder, numStartRoi, numEndRoi)
```

### 2. Extract the coordiantes from ROI files
x1, y1, x2, y2 of every line

```python
def extractCoordinates(measurement_folder_path, numStartRoi, numEndRoi):
    x1Values, y1Values, x2Values, y2Values = [], [], [], []

    # Get ROI files in the current measurement folder and sort them
    roi_files = sorted(
        [f for f in os.listdir(measurement_folder_path) if f.endswith(".roi")],
        key=lambda f: int(f.split(".")[0]),
    )

    for roiFile in roi_files:
        # Extract the number from the file name
        number = int(roiFile.split(".")[0])
        if numStartRoi < number <= numEndRoi:
            roi_path = os.path.join(measurement_folder_path, roiFile)
            roi = read_roi_file(roi_path)

            # Get the ROI coordinates
            for key, value in roi.items():
                x1 = int(value["x1"])
                x2 = int(value["x2"])
                y1 = int(value["y1"])
                y2 = int(value["y2"])

                # Append coordinates to lists
                x1Values.append(x1)
                y1Values.append(y1)
                x2Values.append(x2)
                y2Values.append(y2)

    return x1Values, y1Values, x2Values, y2Values
```

### 3.create lines from the roi lines as a base of polygon

```python
def createLine(x1Values, y1Values, x2Values, y2Values):
    # Initialize points arr for upper and lower line
    xLine = []
    y1Line = []
    y2Line = []

    # Create linear interpolation for upper line
    for i in range(len(x1Values) - 1):
        xSegment = np.linspace(
            x1Values[i], x1Values[i + 1], num=100
        )  # 100 points for linear interpolation
        ySegment = np.linspace(y1Values[i], y1Values[i + 1], num=100)
        xLine.extend(xSegment)
        y1Line.extend(ySegment)

    # Create linear interpolation for lower line
    for i in range(len(x2Values) - 1):
        xSegment = np.linspace(x2Values[i], x2Values[i + 1], num=100)
        ySegment = np.linspace(y2Values[i], y2Values[i + 1], num=100)
        xLine.extend(xSegment)
        y2Line.extend(ySegment)

    return xLine, y1Line, y2Line
```

### 4.Create masks

```python
def generate_mask(roiFolder, numStartRoi, numEndRoi, imagesFolder):
    for foldername in os.listdir(roiFolder):
        # Check if the folder ends with '_measurements'
        if foldername.endswith("_measurements"):
            measurement_folder_path = os.path.join(roiFolder, foldername)
            image_name = foldername.replace("_measurements", ".png")
            img = cv2.imread(os.path.join(imagesFolder, image_name))

            if img is None:
                print(f"Warning: Could not read image {image_name}. Skipping...")
                continue

            imageDimensions = img.shape[:2]
            originalHeight, originalWidth = imageDimensions[0], imageDimensions[1]

            # Create a blank black mask
            mask = np.zeros((originalHeight, originalWidth), dtype=np.uint8)

            # Extract coordinates
            x1Values, y1Values, x2Values, y2Values = extractCoordinates(
                measurement_folder_path, numStartRoi, numEndRoi
            )

            # Remove None values for continuous lines
            x1Values = [x for x in x1Values if x is not None]
            y1Values = [y for y in y1Values if y is not None]
            x2Values = [x for x in x2Values if x is not None]
            y2Values = [y for y in y2Values if y is not None]

            # Create lines and fill the area between them on the mask
            if x1Values and y1Values and x2Values and y2Values:
                xLine, y1Line, y2Line = createLine(
                    x1Values, y1Values, x2Values, y2Values
                )

                # Convert to integer and create polygon points
                points_upper = np.array(
                    list(
                        zip(np.round(xLine).astype(int), np.round(y1Line).astype(int))
                    ),
                    np.int32,
                )
                points_lower = np.array(
                    list(
                        zip(np.round(xLine).astype(int), np.round(y2Line).astype(int))
                    ),
                    np.int32,
                )

                # Combine upper and lower points to create a filled polygon
                points = np.vstack(
                    (points_upper, points_lower[::-1])
                )  # Reverse lower points to close the polygon

                # Fill the area between the upper and lower lines in the mask
                cv2.fillPoly(mask, [points], color=255)

            # Save the mask with the same width and length as the original image
            cv2.imwrite(os.path.join(outputFolder, f"{image_name}.png"), mask)


generate_mask(roiFolder, numStartRoi, numEndRoi, imagesFolder)
```
