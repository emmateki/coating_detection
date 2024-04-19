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

# <center>DATASET creation</center>


## 0. lib

```python
import os
import shutil
import cv2
import random
```

## 1. Put all Data into one final folder and rename them


Need to take all the data from diffrent samples and copy them into one file. Rename them so there are named from 1 to n.

```python

src_folder = "path/to/src/folder" # this conatins folder with subfolders which have folder images and masks and in them samples named with natural numbers
dest_folder_images = "data/Image"
dest_folder_masks = "data/Mask"

train_folder = "DATA_DONE/train"
test_folder = "DATA_DONE/test"
train_ratio = 0.8  # 80% for train


```

```python
def copy_images(src_folder, dest_folder, start_index):
    index = start_index
    for filename in sorted(os.listdir(src_folder)):
        if filename.endswith('.png'):
            src_path = os.path.join(src_folder, filename)
            dest_filename = '{:02d}.png'.format(index)
            dest_path = os.path.join(dest_folder, dest_filename)
            shutil.copy(src_path, dest_path)
            index += 1
```

```python
def copy_into_one():
    if not os.path.exists(dest_folder_images):
        os.makedirs(dest_folder_images)

    if not os.path.exists(dest_folder_masks):
        os.makedirs(dest_folder_masks)

    index_images = 1
    index_masks = 1

    for folder_name in os.listdir(src_folder):
        folder_path = os.path.join(src_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        copy_images(os.path.join(folder_path, "images"), dest_folder_images, index_images)
        copy_images(os.path.join(folder_path, "masks"), dest_folder_masks, index_masks)

        num_images = sum(1 for _ in os.listdir(os.path.join(folder_path, "images"))) 
        num_masks = sum(1 for _ in os.listdir(os.path.join(folder_path, "masks"))) 

        index_images += num_images
        index_masks += num_masks
copy_into_one()

```

# 2. Split data 

```python
def split_data(train_folder, test_folder, train_ratio):
    
    if not os.path.exists(train_folder):
        os.makedirs(os.path.join(train_folder, "train_x"))
        os.makedirs(os.path.join(train_folder, "train_y"))
    if not os.path.exists(test_folder):
        os.makedirs(os.path.join(test_folder, "test_x"))
        os.makedirs(os.path.join(test_folder, "test_y"))
    
    images = sorted(os.listdir(os.path.join(dest_folder_images)))

    num_train = int(len(images) * train_ratio)

    train_images = random.sample(images, num_train)

    for img in train_images:
        src_img_path = os.path.join(dest_folder_images, img)
        dest_img_path = os.path.join(train_folder, "train_x", img)
        shutil.copy(src_img_path, dest_img_path)

        img_num = int(img.split('.')[0])  
        mask = f'{img_num:02d}.png'  
        src_mask_path = os.path.join(dest_folder_masks, mask)
        dest_mask_path = os.path.join(train_folder, "train_y", mask)
        shutil.copy(src_mask_path, dest_mask_path)

    for img in images:
        if img not in train_images:
            src_img_path = os.path.join(dest_folder_images, img)
            dest_img_path = os.path.join(test_folder, "test_x", img)
            shutil.copy(src_img_path, dest_img_path)

            img_num = int(img.split('.')[0])  
            mask = f'{img_num:02d}.png'
            src_mask_path = os.path.join(dest_folder_masks, mask)
            dest_mask_path = os.path.join(test_folder, "test_y", mask)
            shutil.copy(src_mask_path, dest_mask_path)

split_data( train_folder, test_folder, train_ratio)
```

# 3. Check shape

```python
for filename_x in os.listdir(dest_folder_images):
    if filename_x.endswith('.png'):

        x_path = os.path.join(dest_folder_images, filename_x)
        x_img = cv2.imread(x_path, cv2.IMREAD_UNCHANGED)

        filename_y = filename_x 
        y_path = os.path.join(dest_folder_masks, filename_y)
        y_img = cv2.imread(y_path, cv2.IMREAD_UNCHANGED)

        assert x_img.shape == y_img.shape, f"Shape mismatch for {x_path}: {x_img.shape} != {y_path}: {y_img.shape}"

```
