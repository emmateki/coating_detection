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

```python editable=true slideshow={"slide_type": ""}
%load_ext autoreload
%autoreload 2

```

```python editable=true slideshow={"slide_type": ""}
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
import json 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
# Data Preparation
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
data_root = pathlib.Path('/home/tekulova/DATA/data_coating')
csv_roi_path = data_root / 'roi.csv'
```

```python editable=true slideshow={"slide_type": ""}
import scipy.ndimage as ndi
from tqdm.auto import tqdm

# loading the roi measurments into arr


def roiread(image_test_names):  # loading the roi measurments into arr
    with open(csv_roi_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        data_list = list(csv_reader)

    # Sort the rows based on the 'train_name' column
    data_list_sorted = sorted(data_list, key=lambda x: x['train_name'])

    roi_arr = []
    for test_name in image_test_names:
        for row in data_list_sorted:
            if row['train_name'] == test_name:
                original_name = row['original_name']
                train_name = row['train_name']
                roi_file = row['roi_file']
                x1 = row['x1']
                x2 = row['x2']
                y1 = int(row['y1']) if row['y1'].strip(
                ).lower() != 'nan' else np.nan
                y2 = int(row['y2']) if row['y2'].strip(
                ).lower() != 'nan' else np.nan
                length = row['length']

                roi_arr.append([original_name, train_name,
                               roi_file, x1, x2, y1, y2, length])
    return roi_arr


def imread(p):
    img = imageio.imread(p)
    if img.ndim == 3:
        img = img[:, :, 0]

    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = (img - img_min)/(img_max-img_min)

    return np.float32(img_norm)


def imread_mask(p):
    img = imread(p)
    just_mask = np.float32(img > 0)  # ensure only two values 1.0 and 0.0
    return just_mask


def read_set(root, set_name):
    str_set_path = root/f'{set_name}'
    x_root = str_set_path / f"{set_name}_x"

    y_root = str_set_path/f"{set_name}_y"

    x_paths = list(x_root.glob("*.png"))

    # save the names of the samples for roi files
    image_names = [os.path.splitext(p.name)[0] for p in x_paths]

    # ensure the same order
    y_paths = [y_root / p.name for p in x_paths]

    # casting to npfloat
    x_iter = map(imread, x_paths)

    y = tqdm(map(imread_mask, y_paths), total=len(
        x_paths), desc=f"Reading {set_name}")

    # HACK : resizing y to have same dimensions as x
    y_resized = []
    x = []
    for xx, yy in zip(x_iter, y):
        zoom = xx.shape[0]/yy.shape[0], xx.shape[0]/yy.shape[0]
        yy_new = np.float32(ndi.zoom(yy, zoom) == 1)
        assert xx.shape == yy_new.shape, f"{xx.shape=} {yy_new.shape=}"

        half = np.maximum(xx.shape[0]//2, 256)
        y_resized.append(yy_new[:half])
        x.append(xx[:half])

    return x, y_resized, image_names


test_imgs, test_masks, image_test_names = read_set(data_root, 'test')
train_imgs, train_masks, image_train_names = read_set(data_root, 'train')

test_roi = roiread(image_test_names)

assert len(train_imgs) == len(train_masks)
assert len(train_imgs) != 0

assert len(test_imgs) == len(test_masks)
assert len(test_imgs) != 0
print(f"Success {len(train_imgs)=} {len(test_imgs)=}")
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
# Augumentation

Uses albumentation.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
import albumentations as A


def setup_augmentation(
    patch_size,
    crop_or_resize,  # Option: 'crop' or 'resize'
    elastic=False,  # True
    brightness_contrast=True,
    flip_vertical=False,
    flip_horizontal=False,
    blur_sharp_power=None,  # 1
    noise_val=None,  # .01
    rotate_deg=None,  # 90
    interpolation=2,  # constant representing cv2.INTER_CUBIC
):
    transform_list = []
    if crop_or_resize == 'crop':
        patch_size_padded = int(patch_size * 1.5)
        transform_list.append(A.PadIfNeeded(
            patch_size_padded, patch_size_padded))
        transform_list.append(A.CropNonEmptyMaskIfExists(
            height=patch_size, width=patch_size, ignore_values=None, ignore_channels=None))
    elif crop_or_resize == 'resize':
        transform_list.append(
            A.Resize(height=patch_size, width=patch_size, interpolation=interpolation))

    if elastic:
        transform_list.append(
            A.ElasticTransform(
                p=0.5,
                alpha=10,
                sigma=120 * 0.1,
                alpha_affine=120 * 0.1,
                interpolation=interpolation,
            )
        )
    if rotate_deg is not None:
        transform_list.append(
            A.Rotate(limit=rotate_deg, interpolation=interpolation),
        )

    if brightness_contrast:
        transform_list.append(
            A.RandomBrightnessContrast(p=0.5),
        )
    if noise_val is not None:
        transform_list.append(
            A.augmentations.transforms.GaussNoise(noise_val, p=1),
        )

    if blur_sharp_power is not None:
        transform_list.append(
            A.OneOf(
                [
                    A.Sharpen(p=1, alpha=(0.2, 0.2 * blur_sharp_power)),
                    A.Blur(blur_limit=3 * blur_sharp_power, p=1),
                ],
                p=0.3,
            ),
        )

    if flip_horizontal:
        transform_list.append(
            A.HorizontalFlip(p=0.5),
        )
    if flip_vertical:
        transform_list.append(
            A.VerticalFlip(p=0.5),
        )

    transform_list.append(A.CenterCrop(patch_size, patch_size))

    return A.Compose(transform_list)
```

```python editable=true slideshow={"slide_type": ""}
# PATCH SIZE is not the same as resize size! *We do not resize in here.*

def define_transform_fn(patch_size):

    transform_fn = setup_augmentation(
        patch_size,
        crop_or_resize='crop',
        elastic=True,
        brightness_contrast=True,
        flip_vertical=False,  # Oxides never flip vertically
        flip_horizontal=True,
        blur_sharp_power=None,  # hopefully microscopes don't differ in blur much
        noise_val=.01,)

    transform_fn_resize = setup_augmentation(
        patch_size=patch_size, crop_or_resize='resize')

    return transform_fn, transform_fn_resize
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Training

Inspired by tutorial https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

<!-- #endregion -->

### Augumentation of sampels

```python
import albumentations as A
from tqdm import tqdm


def prepare_augmented_dataset(images, masks, transform_fn, color_inversion):
    augmented_images = []
    augmented_masks = []

    for img, mask in tqdm(zip(images, masks), total=len(images)):
        transformed = transform_fn(image=img, mask=mask)

        if color_inversion:
            transformed_image = A.InvertImg(p=0.5)(
                image=transformed["image"])["image"]
        else:
            transformed_image = transformed["image"]

        augmented_images.append(transformed_image)
        augmented_masks.append(transformed["mask"])

    return augmented_images, augmented_masks
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
        assert len(images) == len(labels), f"{len(images)=}!={len(labels)=}"
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

```python editable=true slideshow={"slide_type": ""}
batch_size = 32
train_val_split = .2


def ensure_at_least_batch(data, batch_size):
    return (data*batch_size)[:batch_size]
```

### Adjust Dataset Size and Resize Train/Val Split


```python editable=true slideshow={"slide_type": ""}
def adjust_dataset_size(transform_fn_resize, patch_size, transform_fn):

    augmented_train_images, augmented_train_masks = prepare_augmented_dataset(
        train_imgs, train_masks,  transform_fn, color_inversion=False)
    # ..._rev are augumented pictures with reverse colours
 #   augmented_train_images_rev, augmented_train_masks_rev = prepare_augmented_dataset(train_imgs, train_masks, transform_fn, color_inversion = True)

    train_img_complete = train_imgs + \
        augmented_train_images  # + augmented_train_images_rev
    train_masks_complete = train_masks + \
        augmented_train_masks  # + augmented_train_masks_rev

    assert len(train_img_complete) == len(train_masks_complete)

    val_size = int(len(train_img_complete) * train_val_split)
    train_size = len(train_img_complete) - val_size

    just_train_imgs = train_img_complete[:train_size]
    just_train_masks = train_masks_complete[:train_size]

    val_imgs = train_img_complete[-val_size:]
    val_imgs_res = ensure_at_least_batch(val_imgs, batch_size)

    train_imgs_res = ensure_at_least_batch(just_train_imgs, batch_size)
    train_masks_res = ensure_at_least_batch(just_train_masks, batch_size)

    train_ds = Dataset(
        train_imgs_res,
        train_masks_res,
        transform_fn_resize
    )
    training_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True)

    val_masks = train_masks[-val_size:]
    val_masks_res = ensure_at_least_batch(val_masks, batch_size)

    # There is no augumentation applied on val!
    val_ds = Dataset(
        train_imgs_res,
        train_masks_res,
        transform_fn_resize
    )
    validation_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=32, shuffle=False)

    return training_loader, validation_loader
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

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        bce = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
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

```python editable=true slideshow={"slide_type": ""}
def plot_loss(depth, patch_size, filters):
    for k, v in loss_dict.items():
        plt.plot(v, label=k)

    best_epoch = np.argmin(loss_dict['val_loss'])
    plt.axvline(best_epoch, label=f'{best_epoch=}')
    plt.title("Loss Visualization")
    plt.legend()
    os.makedirs('LossOLD', exist_ok=True)
    plt.savefig(f'LossOLD/loss_{depth}_{patch_size}_{filters}.png')
    plt.close()
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
        x = x[:, :, pad[2]: -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0]: -pad[1]]
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

### Evaluate Roi

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

    y_found_start, y_found_end = find_start_end_positions(
        y_col) if not torch.all(y_col == 0) else (0, 0)

    length_diff = calculate_length_difference(
        y_found_start, y_found_end, int(length))
    start_diff = abs(y1 - y_found_start) if not np.isnan(y1) else np.nan
    end_diff = abs(y2 - y_found_end) if not np.isnan(y2) else np.nan

    return start_diff, end_diff, length_diff


def evaluate_1d(test_roi, inferences):
    infs_trans = [_transform(infc) for infc in inferences]
    start_diffs, end_diffs, length_diffs = [], [], []

    for num, inference in enumerate(infs_trans):
        for i in range(num * 10, num * 10 + 10):
            start_diff, end_diff, length_diff = process_roi_row(
                test_roi[i], inference)
            start_diffs.append(start_diff)
            end_diffs.append(end_diff)
            length_diffs.append(length_diff)

    start_end_diffs = np.concatenate([start_diffs, end_diffs])
    return start_end_diffs, length_diffs
```

```python editable=true slideshow={"slide_type": ""}
def predict(img, model, device, pad_stride=32):
    img_3d = np.stack([img] * 1)
    tensor = torch.from_numpy(img_3d).to(device)[None]
    padded_tensor, pads = pad_to(tensor, pad_stride)
    res_tensor = model(padded_tensor)
    res_unp = unpad(res_tensor, pads)
    # convert to binary mask with this threshold
    res_unp_binary = (res_unp > 0.5).float()
    return res_unp_binary.squeeze(0).squeeze(0)


def eval_save(depth, patch_size, filters, loss):

    with torch.no_grad():
        predictions = [predict(img, model, device) for img in test_imgs]
    # test_masks to binary tensors as well
    mask_arrays = [np.array(mask) for mask in test_masks]
    dtype = torch.float32
    mask_tensors = [torch.tensor((mask > 0.5).astype(
        np.float32), dtype=dtype) for mask in mask_arrays]

    # Evaluate whole area
    metrics = {'IoU': JaccardIndex(task='multiclass', num_classes=2)}
    results = evaluate_2d(mask_tensors, predictions, metrics, device)

    # Evaluate ROI
    start_end_diffs, length_diff = evaluate_1d(test_roi, predictions)

    success = np.sum(~np.isnan(start_end_diffs)) / len(start_end_diffs)
    total_diffs = np.nanmean(start_end_diffs)
    length_diff_total = np.nanmean(length_diff)
    mse_start_end = np.nanmean(np.square(start_end_diffs))
    mse_length = np.nanmean(np.square(length_diff))

    mean_iou = np.mean(results['IoU'])

    data = {
        "Depth": depth,
        "Patch size": patch_size,
        "Filters": filters,
        "Mean IoU": mean_iou,
        "Success of detected lines": success,
        "Total diffs in start and end": total_diffs,
        "Length diff": length_diff_total,
        "MSE of diffs in start and end": mse_start_end,
        "MSE of Length": mse_length,
        "Loss": loss
    }

    serializable_data = {k: convert_to_serializable(
        v) for k, v in data.items()}

    # Load existing data
    try:
        with open('resOLD.json', 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Append new data
    existing_data.append(serializable_data)

    # Save updated data
    with open('resOLD.json', 'w') as f:
        json.dump(existing_data, f, indent=4)


def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
```

### Save predicted masks with roi lines

```python editable=true slideshow={"slide_type": ""}

"""import imageio
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
    
    num += 10 """
```

## Save Pics with Masks


```python editable=true slideshow={"slide_type": ""}
def save(depth, patch_size, filters):
    fol_name = f'results_pics_OLD/Pred_{depth}_{patch_size}_{filters}'
    os.makedirs(fol_name, exist_ok=True)

    with torch.no_grad():
        preds = [predict(img, model, device) for img in test_imgs]

    for i, (img, mask, pred) in enumerate(zip(test_imgs, test_masks, preds)):
        fig, axs = plt.subplots(1, 3, figsize=(21, 7))

        axs[0].imshow(img)
        axs[1].imshow(mask)
        axs[2].imshow(pred)

        plt.savefig(
            f'{fol_name}/plot_{image_test_names[i]}_{depth}_{patch_size}_{filters}.png')
        plt.close(fig)
```

## Training


```python editable=true slideshow={"slide_type": ""}
from tqdm.contrib.logging import logging_redirect_tqdm
import unet
from sklearn.model_selection import ParameterGrid


def loss_wrapper(pred, target_dict):
    return loss_fn(pred, target_dict['y']).mean()


loss_fn = FocalLoss()
"""
param_grid = {
    'depth': [3, 4, 5],
    'patch_size': [64, 128, 256],
    'filters': [8, 16, 32]
}
"""
# Parameter grid for hyperparameter tuning
param_grid = {
    'depth': [5],
    'patch_size': [256],
    'filters': [32]
}

best_val_loss = float('inf')
best_model_state = None

for params in ParameterGrid(param_grid):
    depth = params['depth']
    patch_size = params['patch_size']
    filters = params['filters']

    # Define transformation functions
    transform_fn, transform_fn_resize = define_transform_fn(patch_size)

    # Adjust dataset size
    validation_loader, training_loader = adjust_dataset_size(
        transform_fn_resize, patch_size, transform_fn)

    # Initialize model
    model = unet.UNet(depth=depth, in_channels=1, start_filters=filters)

    # Train model and log progress
    with logging_redirect_tqdm():
        loss_dict = train(
            model,
            training_loader,
            validation_loader,
            loss_wrapper,
            epochs=250,
            device=device
        )

    # Plot loss curves
    plot_loss(depth, patch_size, filters)

    # Evaluate and save model
    eval_save(depth, patch_size, filters, min(loss_dict['val_loss']))
    save(depth, patch_size, filters)

    # Check for best validation loss
    current_val_loss = min(loss_dict['val_loss'])
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_model_state = model.state_dict().copy()

    # Save the best model state
    if best_model_state is not None:
        os.makedirs("models_OLD", exist_ok=True)
        best_model_path = f"models_OLD/best_model_{
            depth}_{patch_size}_{filters}.pth"
        torch.save(best_model_state, best_model_path)
```

```python
exit()
```
