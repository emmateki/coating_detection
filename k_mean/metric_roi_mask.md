---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Corrosion
    language: python
    name: python3
---

# Evaluation of mask with roi lines 


```python
import matplotlib.pyplot as plt
import csv 
import cv2
import os
import pandas as pd
from read_roi import read_roi_file
import numpy as np
```

```python
predicted_mask_folder = ''
roi_folder=""
match_table=""
```

```python
def get_distance(file, number_of_line, x1):
    
    mask=cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape[:2]
    space_between_lines = width / 10 
    x = int(x1)
    if x== 0:
        x = int (space_between_lines / 2 + ( number_of_line * space_between_lines))

    y1 = None
    y2 = None
    for y in range(height):
        pixel_value = mask[y, x]

        if pixel_value == 255:  #white
            if y1 is None:  
                y1 = y
            y2 = y  
    if y1 is None:
        y1=y2=0

    return y1, y2

```

```python

def showMeasurements(roiFolder, img):
    roiFiles = [f for f in os.listdir(roiFolder) if f.endswith('.roi')]
    plt.imshow(img)
    for roiFile in roiFiles:
        number = int(roiFile.split('.')[0])
        if 0 < number <= 10:
            roi_path = os.path.join(roiFolder, roiFile)
            
            if os.path.exists(roi_path): 
                roi = read_roi_file(roi_path)
            
                for key, value in roi.items():
                    xValues = value['x1'], value['x2']
                    yValues = value['y1'], value['y2']

                    plt.plot(xValues, yValues, 'r')
        
    plt.axis('off')
    plt.show()


def extract_roi_info(roi_folder_path, number_of_line):
    roi_file_path = os.path.join(roi_folder_path, f"{number_of_line}.roi")

    if os.path.exists(roi_file_path):
        roi = read_roi_file(roi_file_path)
        y1, y2, x1 = extract_y_values_from_roi(roi)
        roi_length = y2 - y1
        return y1, y2, roi_length ,x1
    else:
        return 0, 0, 0, 0 

def extract_y_values_from_roi(roi):
    for key, value in roi.items():
        y2 = value['y1']
        y1 = value['y2']
        x1 = value['x1']
        x2= value['x2']
    return y1,y2,x1

```

```python
def process_files(predicted_mask_folder, roi_folder ):

    y1_pred_arr = []
    y2_pred_arr = []
    y1_roi_arr = []
    y2_roi_arr = []
    length_pred_arr = []
    length_roi_arr = []

    for filename_pred in os.listdir(predicted_mask_folder):
        if filename_pred.endswith('.png'):

            for number_of_line in range(10):

                csvFile = pd.read_csv(match_table)

                if filename_pred in csvFile.iloc[:, 0].values:
                    index = csvFile.index[csvFile.iloc[:, 0] == filename_pred].tolist()[0]
                    roi_folder_name = csvFile.iloc[index, 1]
                    base, extension = os.path.splitext(roi_folder_name)
                    roi_folder_name = base +'_measurements'


                roi_start=roi_end=roi_length = 0;
                roi_folder_path = os.path.join(roi_folder, roi_folder_name)
                file_path_pred= os.path.join(predicted_mask_folder, filename_pred)



                y1_roi, y2_roi, length_roi, x1 = extract_roi_info(roi_folder_path, number_of_line+1)
                y1_pred, y2_pred = get_distance(file_path_pred,number_of_line ,x1)

                y1_pred_arr.append(y1_pred)
                y2_pred_arr.append(y2_pred)
                y1_roi_arr.append(y1_roi)
                y2_roi_arr.append(y2_roi)

                length_pred_arr.append (y2_pred - y1_pred)
                length_roi_arr.append (length_roi)
    

        
        mask1 = cv2.imread(file_path_pred, cv2.IMREAD_GRAYSCALE)
        showMeasurements(roi_folder_path, mask1)

    return y1_pred_arr, y2_pred_arr, y1_roi_arr, y2_roi_arr, length_pred_arr, length_roi_arr


y1_pred, y2_pred, y1_roi, y2_roi, length_pred, length_roi= process_files(predicted_mask_folder, roi_folder )

```

# Statistics

```python
y1_pred = np.array(y1_pred)
y2_pred = np.array(y2_pred)
y1_roi = np.array(y1_roi)
y2_roi = np.array(y2_roi)
length_pred = np.array(length_pred)
length_roi = np.array(length_roi)

iouLine =[]

#MSE
mse_start = np.mean((y1_pred - y1_roi) ** 2)
mse_end = np.mean((y2_pred - y2_roi) ** 2)
mse_length = np.mean((length_pred - length_roi) ** 2)

#MAE
mae_start = np.mean(np.abs(y1_pred - y1_roi))
mae_end = np.mean(np.abs(y2_pred - y2_roi))
mae_length = np.mean(np.abs(length_pred - length_roi))


for i in range(len(y1_pred)):
    intersection = min(y2_pred[i], y2_roi[i]) - max(y1_pred[i], y1_roi[i])
    union = max(y2_pred[i], y2_roi[i]) - min(y1_pred[i], y1_roi[i])
    iouLine.append(intersection / union)


#IoU lines
mean_IoU_lines = np.mean(iouLine)


```

```python
with open('statistics_roi.txt', 'w') as f:
    f.write("MSE:\n")
    f.write(f"Start: {mse_start:.6f}\n")
    f.write(f"End: {mse_end:.6f}\n")
    f.write(f"Length: {mse_length:.6f}\n\n")

    f.write("MAE:\n")
    f.write(f"Start: {mae_start:.6f}\n")
    f.write(f"End: {mae_end:.6f}\n")
    f.write(f"Length: {mae_length:.6f}\n\n")

    f.write("IoU:\n")
    f.write(f"Lines: {mean_IoU_lines:.6f}\n")
```

```python

tolerances = range(7)

exact_matches = []

total_lines = len(length_pred)

for tolerance in tolerances:
    exact_matches_length = (np.abs(length_pred - length_roi) <= tolerance).sum()
    percentage_exact_matches_length_kmeans = (exact_matches_length / total_lines) * 100
    exact_matches.append(percentage_exact_matches_length_kmeans)

plt.plot(tolerances, exact_matches, label='prediction')

plt.xlabel('tolerance(pixels)')
plt.ylabel('exact matches (%)')
plt.title('Percentage of exact matches for lengths with different tolerances')
plt.legend()


plt.grid(True)
plt.show()

```
