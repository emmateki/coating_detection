---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2

```

```python
import pathlib
import imageio
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchmetrics
from torchmetrics import JaccardIndex
import csv
import os
import cv2

device = 'cpu' # if cuda available
```

# Data Preparation

```python
data_root = pathlib.Path('/DATA/data_coating')
```

```python
import scipy.ndimage as ndi
from tqdm.auto import tqdm 

#NEW function for loading the roi measurments into arr
def roiread(image_test_names):
    csv_roi_path = data_root / 'roi.csv'
    roi_arr = []
    
    for image_test_name in image_test_names:
        with open(csv_roi_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)  # Skip first row - column names
            
            for row in csv_reader:
                if row[1] == image_test_name:

                    original_name = row[0]
                    train_name = row[1]
                    roi_file = row[2]
                    x1 = row[3]
                    x2 = row[4]
                    y1 = int(row[5]) if row[5].strip().lower() != 'nan' else np.nan
                    y2 = int(row[6]) if row[6].strip().lower() != 'nan'  else np.nan
                    length = row[7]
                    
                    roi_arr.append([original_name, train_name, roi_file, x1, x2, y1, y2, length])
    return roi_arr
    
def imread(p):
    img = imageio.imread(p)
    if img.ndim == 3:
        img = img[:,:,0]
    
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm= (img - img_min)/(img_max-img_min)
    
    return np.float32(img_norm)


def imread_mask(p):
    img = imread(p)
    just_mask = np.float32(img > 0) # ensure only two values 1.0 and 0.0    
    return just_mask

def read_set(root,set_name):
    str_set_path = root/f'{set_name}'
    x_root = str_set_path/ f"{set_name}_x"
    #EDIT THIS
    y_root = str_set_path/f"{set_name}_y_without_oxidation"
    
    x_paths = list(x_root.glob("*.png"))

    # NEW save the names of the samples for roi files
    image_names = [os.path.splitext(p.name)[0] for p in x_paths]

    # ensure the same order
    y_paths = [y_root / p.name for p in x_paths]

    # casting to npfloat
    x_iter = map(imread,x_paths)
    
    y = tqdm(map(imread_mask,y_paths),total = len(x_paths),desc = f"Reading {set_name}")

    # HACK : resizing y to have same dimensions as x
    y_resized = []
    x = []
    for xx,yy in zip(x_iter,y):   
        zoom = xx.shape[0]/yy.shape[0],xx.shape[0]/yy.shape[0]
        yy_new = np.float32(ndi.zoom(yy,zoom) == 1)
        assert xx.shape == yy_new.shape,f"{xx.shape=} {yy_new.shape=}"
        
        half = np.maximum(xx.shape[0]//2, 256)
        y_resized.append(yy_new[:half])
        x.append(xx[:half])
        
    return x,y_resized, image_names

test_imgs,test_masks, image_test_names = read_set(data_root, 'test')
train_imgs,train_masks,image_train_names = read_set(data_root, 'train')

test_roi = roiread(image_test_names)

assert len(train_imgs) == len(train_masks)
assert len(test_imgs) == len(test_masks)
```

# Augumentation

Uses albumentation.

```python editable=true slideshow={"slide_type": ""}
import albumentations as A

def setup_augmentation(
    patch_size,
    crop_or_resize,  # Option: 'crop' or 'resize'
    elastic=False,  # True
    brightness_contrast=False,
    flip_vertical=False,
    flip_horizontal=False,
    blur_sharp_power=None,  # 1
    noise_val=None,  # .01
    rotate_deg=None,  # 90
    interpolation=2, # constant representing cv2.INTER_CUBIC
):
    transform_list = []
    if crop_or_resize == 'crop':
        patch_size_padded = int(patch_size * 1.5)
        transform_list.append(A.PadIfNeeded(patch_size_padded, patch_size_padded))
        transform_list.append(A.RandomCrop(patch_size, patch_size))
    elif crop_or_resize == 'resize':
        transform_list.append(A.Resize(height=patch_size, width=patch_size, interpolation=interpolation))


    if elastic:
        transform_list += [
            A.ElasticTransform(
                p=0.5,
                alpha=10,
                sigma=120 * 0.1,
                alpha_affine=120 * 0.1,
                interpolation=interpolation,
            )
        ]
    if rotate_deg is not None:
        transform_list += [
            A.Rotate(limit=rotate_deg, interpolation=interpolation),
        ]

    if brightness_contrast:
        transform_list += [
            A.RandomBrightnessContrast(p=0.5),
        ]
    if noise_val is not None:
        transform_list += [
            A.augmentations.transforms.GaussNoise(noise_val, p=1),
        ]

    if blur_sharp_power is not None:
        transform_list += [
            A.OneOf(
                [
                    A.Sharpen(p=1, alpha=(0.2, 0.2 * blur_sharp_power)),
                    A.Blur(blur_limit=3 * blur_sharp_power, p=1),
                ],
                p=0.3,
            ),
        ]

    if flip_horizontal:
        transform_list += [
            A.HorizontalFlip(p=0.5),
        ]
    if flip_vertical:
        transform_list += [
            A.VerticalFlip(p=0.5),
        ]

    transform_list += [A.CenterCrop(patch_size, patch_size)]
    return A.Compose(transform_list)
```

```python
# PATCH SIZE is not the same as resize size! *We do not resize in here.*
patch_size = 256 # or more?

transform_fn = setup_augmentation(
    patch_size,
    crop_or_resize='crop',
    elastic = True,
    brightness_contrast=False,
    flip_vertical=False, # Oxides never flip vertically
    flip_horizontal=True,
    blur_sharp_power=None, # hopefully microscopes don't differ in blur much
    noise_val=.01,
    rotate_deg=None,
)
transform_fn_resize = setup_augmentation(patch_size=256, crop_or_resize='resize')


for x,y in zip(train_imgs + test_imgs,train_masks + test_masks):
    x=np.float32(x)
    transformed = transform_fn(image=x, mask=y)
    tr_x = transformed["image"]
    tr_y = transformed["mask"]

    _,axs = plt.subplots(1,2)
    axs[0].imshow(tr_x,vmin=0,vmax=1)
    axs[1].imshow(tr_y)
    plt.show()
    break
    
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Training

Inspired by tutorial https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

<!-- #endregion -->

### Augumentation of sampels

```python
def prepare_augmented_dataset(images, masks, transform_fn):

    augmented_images = []
    augmented_masks = []

    for img, mask in tqdm(zip(images, masks), total=len(images)):
        
        transformed = transform_fn(image=img, mask=mask)
    
        augmented_images.append(transformed["image"])
        augmented_masks.append(transformed["mask"])

    return augmented_images, augmented_masks

```

```python

augmented_train_images, augmented_train_masks = prepare_augmented_dataset(train_imgs, train_masks, transform_fn)

train_img_complete = train_imgs + augmented_train_images
train_masks_complete = train_masks + augmented_train_masks

assert len(train_img_complete) == len(train_masks_complete)

```

```python
for x,y in zip(augmented_train_images,augmented_train_masks):
    _,axs = plt.subplots(1,2)
    axs[0].imshow(x,vmin=0,vmax=1)
    axs[1].imshow(y)
    plt.show()
    break

```

## Dataset

```python editable=true slideshow={"slide_type": ""}
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        images, 
        labels, 
        transform, 
    ):
        assert len(images) == len(labels),f"{len(images)=}!={len(labels)=}"
        self.images = images
        self.labels = labels
        self.transform = transform        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        transformed = transform_fn(image=image, mask=label)
        tr_x = transformed["image"]
        tr_y = transformed["mask"]
    
        return {
            "x": tr_x[None],
            "y": tr_y[None],
        }

```

### Train/Val Split


```python editable=true slideshow={"slide_type": ""}
batch_size = 32
train_val_split = .2

def ensure_at_least_batch(data,batch_size):
    return (data*batch_size)[:batch_size]

val_size = int(len(train_img_complete) * train_val_split)
train_size = len(train_img_complete) - val_size

just_train_imgs = train_img_complete[:train_size]
just_train_masks = train_masks_complete[:train_size]

val_imgs = train_img_complete[-val_size:]
val_imgs_res = ensure_at_least_batch(val_imgs,batch_size)


```

### Adjust Dataset Size and Resize


```python

train_imgs_res = ensure_at_least_batch( just_train_imgs,batch_size)
train_masks_res = ensure_at_least_batch(just_train_masks,batch_size)

train_ds = Dataset(
    train_imgs_res,
    train_masks_res,
    transform_fn_resize
)
training_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

val_masks = train_masks[-val_size:]
val_masks_res = ensure_at_least_batch(val_masks,batch_size)

# There is no augumentation applied on val!
val_ds = Dataset(
    train_imgs_res,
    train_masks_res,
    transform_fn_resize
)
validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

```

```python

'''
import numpy as np
import matplotlib.pyplot as plt

def plot_images(dataset, num_images=10):
    """
    Plot a sample of images from the dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset containing images.
        num_images (int): Number of images to plot.
    """
    plt.figure(figsize=(15, 8))
    rows = (num_images // 5) + 1  # Calculate number of rows for subplots
    
    for i in range(num_images):
        sample = dataset[i]  # Get a sample from the dataset
        
        # Access image and mask using correct keys
        image = sample['x']  # Assuming 'image' is the key for the image
        mask = sample['y']    # Assuming 'mask' is the key for the mask
        
        # Convert PyTorch tensor to NumPy array if necessary
        if isinstance(image, np.ndarray):
            image = image  # NumPy array
        else:
            image = image.squeeze().numpy()  # PyTorch tensor to NumPy array
        
        # Handle different image shapes
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]  # Remove batch dimension for grayscale images
        elif image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)  # Reshape for RGB images
        
        plt.subplot(rows, 5, i + 1)  # Create subplot
        plt.imshow(image)  # Display image
        plt.title(f"Sample {i + 1}")  # Set subplot title
        plt.axis('off')  # Turn off axis
    
    plt.tight_layout()
    plt.show()

# Assuming `train_dataset` is your training dataset
plot_images(train_ds, num_images=32)  # Plot the first 32 images from the dataset
'''

```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Model

See unet.py for more detail.
<!-- #endregion -->

```python
import unet
model = unet.UNet(depth = 4,in_channels=1, start_filters=8)
```

## Loss Function

```python
from torch.functional import F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs,targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        bce = F.binary_cross_entropy(inputs,targets, reduction=self.reduction)
        bce_exp = torch.exp(-bce)
        return self.alpha * (1 - bce_exp) ** self.gamma * bce

```

# Define training

- epochs
  - training
    - steps
  - validation
    - steps

```python
import logging 
logging.basicConfig()
logger = logging.getLogger("training")
logger.setLevel(logging.DEBUG)
def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    epochs=0,
    lr=0.001,
    device="cpu",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def step_fn(targets):
        return step(model, targets, loss_fn, device=device)

    def train_epoch_fn():
        return train_epoch(model, train_dataloader, optimizer, step_fn)

    def eval_epoch_fn():
        return validate_epoch(model, val_dataloader, step_fn)

    train_losses = []
    validation_losses = []
    epochs_iter = range(epochs) if epochs is not None else itertools.count()
    for epoch in tqdm(epochs_iter, desc="Training epochs"):
        loss_train, loss_val = run_epoch(
            train_epoch_fn, eval_epoch_fn
        )
        train_losses.append(loss_train)
        validation_losses.append(loss_val)

        logger.info(f"{epoch=} {loss_val=:.5f}")

    return {"train_loss": train_losses, "val_loss": validation_losses}

def run_epoch(
    train_epoch_fn,
    validate_epoch_fn
):
    train_loss = train_epoch_fn()
    val_loss = validate_epoch_fn()

    return train_loss, val_loss

def train_epoch(model, dataloader, optimizer, step_fn):
    model.train()

    def train_step(targets):
        optimizer.zero_grad()
        ls = step_fn(targets)
        ls.backward()
        optimizer.step()
        return ls.item()

    losses = [train_step(targets) for targets in dataloader]
    return np.mean(losses)


def validate_epoch(model, dataloader, step_fn):
    model.eval()
    with torch.no_grad():
        losses = [step_fn(t).item() for t in dataloader]
        return np.mean(losses)


def step(model, targets, loss_fn, device="cpu"):
    device_targets = {k: v.to(device) for k, v in targets.items()}
    pred = model(device_targets["x"])
    return loss_fn(pred, device_targets)


```

```python
from tqdm.contrib.logging import logging_redirect_tqdm

loss_fn = FocalLoss()
def loss_wrapper(pred,target_dict):
    return loss_fn(pred,target_dict['y']).mean()

with logging_redirect_tqdm():
    loss_dict = train(
        model,
        training_loader,
        validation_loader,
        loss_wrapper,
        epochs = 250,
        device =device
    )
```

```python
for k,v in loss_dict.items():
    plt.plot(v,label=k)

best_epoch = np.argmin(loss_dict['val_loss'])
plt.axvline(best_epoch,label=f'{best_epoch=}')
plt.title("Loss Visualization")
plt.legend()
```

# Evaluation

```python
def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2] : -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0] : -pad[1]]
    return x
```

```python
def _transform(inference):
    # remove first and second dimension
    transformed_inference = inference.squeeze(0).squeeze(0)
    return transformed_inference

def _is_valid_shape(gt, infc):
    assert gt.shape == infc.shape, "Both tensors must have the same shape."

def evaluate_2d(ground_truths, inferences, metrics, device):
    gts = [gt.to(device) for gt in ground_truths]
    infs_trans = [_transform(infc) for infc in inferences]

    for gt, infc in zip(gts, infs_trans):
        _is_valid_shape(gt, infc)

    eval_results = {}
    for name, metric in metrics.items():
        m_result = [metric(gt, infc) for gt, infc in zip(gts, infs_trans)]
        eval_results[name] = m_result

    return eval_results

```

```python
def find_start_end_positions(y_col):
    max_y = y_col.shape[0]
    y_found_start = np.argmax(y_col == 1)
    y_found_end = max_y - 1 - np.argmax(torch.flip(y_col, dims=(0,)) == 1)
    return y_found_start, y_found_end

def calculate_length_difference(y_found_start, y_found_end, length):
    return abs(abs(y_found_end - y_found_start) - length)

def process_roi_row(roi_row, inference):
    x1, x2, y1, y2, length = map(float, roi_row[3:8])
    y_col = inference[:, int(x1)]
    
    y_found_start, y_found_end = find_start_end_positions(y_col) if not torch.all(y_col == 0) else (0, 0)
    
    length_diff = calculate_length_difference(y_found_start, y_found_end, int(length))
    start_diff = abs(y1 - y_found_start) if not np.isnan(y1) else np.nan
    end_diff = abs(y2 - y_found_end) if not np.isnan(y2) else np.nan
    
    return start_diff, end_diff, length_diff

def evaluate_1d(test_roi, inferences):
    infs_trans = [_transform(infc) for infc in inferences]
    start_diffs, end_diffs, length_diffs = [], [], []

    for num, inference in enumerate(infs_trans):
        for i in range(num * 10, num * 10 + 10):
            start_diff, end_diff, length_diff = process_roi_row(test_roi[i], inference)
            start_diffs.append(start_diff)
            end_diffs.append(end_diff)
            length_diffs.append(length_diff)

    start_end_diffs = np.concatenate([start_diffs, end_diffs])
    return start_end_diffs, length_diffs
```

```python
def predict(img, model, device, pad_stride=32):
    img_3d = np.stack([img] * 1)
    tensor = torch.from_numpy(img_3d).to(device)[None]
    padded_tensor, pads = pad_to(tensor, pad_stride)
    res_tensor = model(padded_tensor)
    res_unp = unpad(res_tensor, pads)
    # convert to binary mask with this threshold 
    res_unp_binary = (res_unp > 0.5).float()  
    return res_unp_binary.squeeze(0).squeeze(0)


with torch.no_grad():
    predictions = [predict(img, model, device) for img in test_imgs]

# test_masks to binary tensors as wel
mask_arrays = [np.array(mask) for mask in test_masks]
dtype = torch.float32
mask_tensors = [torch.tensor((mask > 0.5).astype(np.float32), dtype=dtype) for mask in mask_arrays]

#Eval whole area
metrics = {'IoU': JaccardIndex(task='multiclass', num_classes=2)}
results = evaluate_2d(mask_tensors, predictions, metrics, device)

# evaluate roi
start_end_diffs, length_diff = evaluate_1d(test_roi, predictions)

success = np.sum(~np.isnan(start_end_diffs))/len(start_end_diffs)
total_difs = np.nanmean(start_end_diffs)
length_diff_total = np.nanmean(length_diff)
mse_start_end = np.mean(np.square(start_end_diffs))
mse_length = np.mean(np.square(length_diff))


print("Mean IoU:", np.mean(results['IoU']))
print("Succes of detected lines", success)
print("Total diffs in start and end", total_difs)
print("Length diff", length_diff_total)
print("MSE of diffs in start and end", mse_start_end)
print("MSE of Length ", mse_length)


```

### Save predicted masks with roi lines

```python editable=true slideshow={"slide_type": ""}
import imageio
import os

save_dir = "results"

os.makedirs(save_dir, exist_ok=True)

num = 0
for prediction_index, prediction in enumerate(predictions):
    first_test_img = np.squeeze(prediction)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(first_test_img, cmap='gray')
    ax.axis('off')  

    # Plot the ROIs on the image
    for i in range(num, num + 10):
        first_roi_row = test_roi[i]
        x1, x2, y1, y2 = map(float, first_roi_row[3:7])  

        if not (np.isnan(y1) or np.isnan(y2)):
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
    
    # Save the plot as an image with a transparent background
    save_path = os.path.join(save_dir, f"prediction_{prediction_index}.png")
    fig.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)  
    plt.close() 
    
    num += 10 

```

```python editable=true slideshow={"slide_type": ""}

with torch.no_grad():
    preds = [predict(img, model, device) for img in test_imgs]

for imgs in zip(test_imgs,test_masks, preds):
    fig,axs = plt.subplots(1,3,figsize=(21,7))
    _=[ax.imshow(img) for ax,img in zip(axs,imgs)]
    plt.show()
```

```python
exit ()
```
