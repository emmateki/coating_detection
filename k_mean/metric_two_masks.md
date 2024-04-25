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

# Evaluation of mask with template mask 


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
template_mask_folder = ''

```

```python
def get_distance(file, number_of_line):
    
    mask=cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape[:2]
    space_between_lines = width / 10 

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
def overlay_masks_and_IoU(mask1, mask2):
    mask1_rgb = cv2.cvtColor(mask1, cv2.COLOR_GRAY2RGB)
    mask2_rgb = cv2.cvtColor(mask2, cv2.COLOR_GRAY2RGB)

    mask1_rgb = cv2.bitwise_not(mask1_rgb)
    mask2_rgb = cv2.bitwise_not(mask2_rgb)

    #  black regions in mask1 to red
    mask1_rgb[mask1_rgb[:, :, 0] == 0] = [0, 0, 255] 

    # black regions in mask2 to blue
    mask2_rgb[mask2_rgb[:, :, 0] == 0] = [255, 0, 0]  

    overlay = cv2.addWeighted(mask1_rgb, 0.5, mask2_rgb, 0.5, 0)

    # Calculate IoU
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)
    iou = cv2.countNonZero(intersection) / cv2.countNonZero(union)


    plt.imshow(overlay)
    plt.axis('off')
    plt.title(f'Overlay of masks with IOU: {iou:.2f}')
    plt.show()

    return iou

```

```python
def process_files(predicted_mask_folder, template_mask_folder ):

    y1_pred_arr = []
    y2_pred_arr = []
    y1_temp_arr = []
    y2_temp_arr = []
    length_pred_arr= []
    length_temp_arr = []
    iou = []

    for filename_pred in os.listdir(predicted_mask_folder):
        if filename_pred.endswith('.png'):
            file_path_temp= os.path.join(template_mask_folder, filename_pred)
            file_path_pred= os.path.join(predicted_mask_folder, filename_pred)

            for number_of_line in range(10):
            
                y1_pred, y2_pred = get_distance(file_path_pred,number_of_line)
                y1_temp, y2_temp = get_distance(file_path_temp, number_of_line )

                y1_pred_arr.append(y1_pred)
                y2_pred_arr.append(y2_pred)
                y1_temp_arr.append(y1_temp)
                y2_temp_arr.append(y2_temp)
                

                length_pred_arr.append (y2_pred - y1_pred)
                length_temp_arr.append (y2_temp - y1_temp)


        
        mask1 = cv2.imread(file_path_temp, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(file_path_pred, cv2.IMREAD_GRAYSCALE)
        iou.append (  overlay_masks_and_IoU(mask1, mask2))
    return y1_pred_arr, y2_pred_arr, y1_temp_arr, y2_temp_arr, length_pred_arr, length_temp_arr, iou


y1_pred, y2_pred, y1_temp, y2_temp, length_pred, length_temp, iou = process_files(predicted_mask_folder, template_mask_folder )

```

# Statistics

```python
y1_pred = np.array(y1_pred)
y2_pred = np.array(y2_pred)
y1_temp = np.array(y1_temp)
y2_temp = np.array(y2_temp)
length_pred = np.array(length_pred)
length_temp = np.array(length_temp)
iou = np.array(iou)
iouLine =[]

#MSE
mse_start = np.mean((y1_pred - y1_temp) ** 2)
mse_end = np.mean((y2_pred - y2_temp) ** 2)
mse_length = np.mean((length_pred - length_temp) ** 2)

#MAE
mae_start = np.mean(np.abs(y1_pred - y1_temp))
mae_end = np.mean(np.abs(y2_pred - y2_temp))
mae_length = np.mean(np.abs(length_pred - length_temp))

#IoU whoel Area
mean_IoU_whole_area = np.sum(iou) / len(iou)


for i in range(len(y1_pred)):
    intersection = min(y2_pred[i], y2_temp[i]) - max(y1_pred[i], y1_temp[i])
    union = max(y2_pred[i], y2_temp[i]) - min(y1_pred[i], y1_temp[i])
    iouLine.append(intersection / union)


#IoU lines
mean_IoU_lines = np.mean(iouLine)

#IoU combined 
mean_IoU_combined = (mean_IoU_whole_area + mean_IoU_lines) / 2

```

```python
with open('statistics_masks.txt', 'w') as f:
    f.write("MSE:\n")
    f.write(f"Start: {mse_start:.6f}\n")
    f.write(f"End: {mse_end:.6f}\n")
    f.write(f"Length: {mse_length:.6f}\n\n")

    f.write("MAE:\n")
    f.write(f"Start: {mae_start:.6f}\n")
    f.write(f"End: {mae_end:.6f}\n")
    f.write(f"Length: {mae_length:.6f}\n\n")

    f.write("IoU:\n")
    f.write(f"Whole Area: {mean_IoU_whole_area:.6f}\n")
    f.write(f"Lines: {mean_IoU_lines:.6f}\n")
    f.write(f"Combined: {mean_IoU_combined:.6f}\n")
```

```python

tolerances = range(7)

exact_matches = []

total_lines = len(length_pred)

for tolerance in tolerances:
    exact_matches_length = (np.abs(length_pred - length_temp) <= tolerance).sum()
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
