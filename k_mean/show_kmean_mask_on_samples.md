---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
---

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


image_folder = "path"
mask_folder = "path"

def show(image_folder,mask_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):  

            image_path = os.path.join(image_folder, filename)
            original_image = cv2.imread(image_path)

            mask_path = os.path.join(mask_folder, filename)
            binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
            # Convert the binary mask to 3 channels
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            
            # Apply red color to the binary mask
            binary_mask[np.where((binary_mask == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
            # red mask half-transparent
            alpha = 0.5
            overlay = cv2.addWeighted(binary_mask, alpha, original_image, 1 - alpha, 0)

            plt.figure()
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f'Sample with mask: {filename}')
            plt.show()

            #if save uncomment

            #image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            #mask_Path = os.path.join( 'path' + filename)
            #cv2.imwrite(mask_Path, image)

show(image_folder,mask_folder)
```
