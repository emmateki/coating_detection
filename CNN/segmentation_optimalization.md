---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python editable=true slideshow={"slide_type": ""}
%load_ext autoreload
%autoreload 2
```

# Setup and Configuration

all paths are defined in config.json file 

```python editable=true slideshow={"slide_type": ""}
import os
import csv
import copy
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import scipy.ndimage as ndi
import imageio
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import optuna
import json
import itertools

from torch.functional import F

# Segmentation model
from segmentation_models_pytorch import Unet

# Custom modules and functions
from plots_CNN import plot_predictions, plot_loss, plot_histogram
from loss_functions import IoULoss
from evaluation import eval_save
from augumentation import setup_augmentation

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# IMPORTNAT NOTE: config.json must be in the same directory as this script

with open("config.json", "r") as f:
    config = json.load(f)

idx = str(config["paths"]["idx"])


def update_paths(paths, idx):
    for key, path in paths.items():
        if isinstance(path, str):
            paths[key] = path.replace("{idx}", idx)
    return paths


updated_paths = update_paths(config["paths"], idx)

eval_results_path = Path(config["paths"]["eval_results_path"])
model_path = Path(config["paths"]["model_path"])
predictions_path = Path(config["paths"]["predictions_path"])
loss_path = Path(config["paths"]["loss_path"])
histogram_path = Path(config["paths"]["histogram_path"])

batch_size = config["model"]["batch_size"]
# epochs = config["model"]["epochs"]
epochs = 200

data_root = Path(config["paths"]["data_root"])
csv_roi_path = Path(config["paths"]["csv_roi_path"])
```

```python
seed = 678


def set_seed(seed=678):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(678)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
# Data Preparation

<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
def roi_read(image_test_names):
    """
    Reads ROI (Region of Interest) measurements from a CSV file and filters them based on given image test names and then save data into array.

    Reads contents of csv file into a list - dictionary, sorts this by the 'train_name' column (for quicker itertion),
    and then iterates over the sorted list to find and reformat ROI data for each specified test name in 'image_test_names'.
    If a match is found, it collects the ROI data, including handling 'nan' values for 'y1' and 'y2' by converting them to numpy.nan.

    Parameters:
    - image_test_names (list of str): A list of test names to filter the ROI data by.

    Returns:
    - list of lists: A list containing the filtered ROI data. Each element of the list is another list with the following elements:
        - original_name (str): The original name of the image.
        - train_name (str): The training name associated with the image.
        - roi_file (str): The file name of the ROI. (01,02,...)
        - x1 (str): The x-coordinate of the top of the line of the ROI.
        - x2 (str): The x-coordinate of the bottom of the line of the ROI.
        - y1 (int or numpy.nan): The y-coordinate of the top of the line of the ROI, converted to an integer or numpy.nan if 'nan'.
        - y2 (int or numpy.nan): The y-coordinate of the bottom of the line corner of the ROI, converted to an integer or numpy.nan if 'nan'.
        - length (str): The length of the ROI. y2-y1.
    """
    roi_arr = []
    with open(csv_roi_path, "r") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        data_list = list(csv_reader)

    # cast image_test_names to integers
    names_set = [int(name) for name in image_test_names]
    # Create a dictionary to map train_name (as integer) to its index in names_set
    name_to_index = {name: index for index, name in enumerate(names_set)}
    # Sort data_list based on the order in image_test_names (or names_set)
    data_list_sorted = sorted(
        data_list, key=lambda x: name_to_index.get(
            int(x["train_name"]), float("inf"))
    )
    the_good_rows = (r for r in data_list_sorted if int(
        r["train_name"]) in names_set)

    for row in the_good_rows:
        original_name = row["original_name"]
        train_name = row["train_name"]
        roi_file = row["roi_file"]
        x1 = row["x1"]
        x2 = row["x2"]
        y1 = int(row["y1"]) if row["y1"].strip().lower() != "nan" else np.nan
        y2 = int(row["y2"]) if row["y2"].strip().lower() != "nan" else np.nan
        length = row["length"]

        roi_arr.append([original_name, train_name,
                       roi_file, x1, x2, y1, y2, length])
    return roi_arr


def imread(p):
    img = imageio.imread(p)
    if img.ndim == 3:
        img = img[:, :, 0]

    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = (img - img_min) / (img_max - img_min)

    return np.float32(img_norm)


def imread_mask(p):
    img = imread(p)
    just_mask = np.float32(img > 0)
    return just_mask


def read_set(root, set_name):
    str_set_path = root / f"{set_name}"
    x_root = str_set_path / f"{set_name}_x"

    y_root = str_set_path / f"{set_name}_y"

    x_paths = list(x_root.glob("*.png"))

    # save the names of the samples for roi files
    image_names = [os.path.splitext(p.name)[0] for p in x_paths]

    # ensure the same order
    y_paths = [y_root / p.name for p in x_paths]
    x_iter = map(imread, x_paths)

    y = tqdm(map(imread_mask, y_paths), total=len(
        x_paths), desc=f"Reading {set_name}")

    # HACK : resizing y to have same dimensions as x
    y_resized = []
    x = []
    for xx, yy in zip(x_iter, y):
        zoom = xx.shape[0] / yy.shape[0], xx.shape[0] / yy.shape[0]
        yy_new = np.float32(ndi.zoom(yy, zoom) == 1)
        assert xx.shape == yy_new.shape, f"{xx.shape=} {yy_new.shape=}"

        half = np.maximum(xx.shape[0] // 2, 256)
        y_resized.append(yy_new[:half])
        x.append(xx[:half])

    return x, y_resized, image_names


test_imgs, test_masks, image_test_names = read_set(data_root, "test")
train_imgs, train_masks, image_train_names = read_set(data_root, "train")

test_roi = roi_read(image_test_names)

assert len(train_imgs) == len(train_masks)
assert len(train_imgs) != 0

assert len(test_imgs) == len(test_masks)
assert len(test_imgs) != 0
print(f"Success {len(train_imgs)=} {len(test_imgs)=}")
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
# Augumentation

<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
def define_transform_fn(patch_size):
    transform_fn = setup_augmentation(
        patch_size,
        crop=True,
        elastic=True,
        brightness_contrast=True,
        flip_horizontal=True,
        gauss_noise=True,
        blur_sharp=True,
    )

    transform_fn_crop = setup_augmentation(patch_size, crop=True)

    return transform_fn, transform_fn_crop
```

## Dataset and loader

```python
# Code inspired by Jaroslav Knotek
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
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            tr_x = transformed["image"]
            tr_y = transformed["mask"]
        else:
            tr_x = image
            tr_y = label

        return {
            "x": tr_x[None],
            "y": tr_y[None],
        }
```

```python
def loader(patch_size):

    num_samples = len(train_imgs)

    # Randomly select unique indices from the training dataset
    random_indices = np.random.choice(
        num_samples, int(num_samples*0.25), replace=False)

    val_imgs = [train_imgs[i] for i in random_indices]
    val_masks = [train_masks[i] for i in random_indices]

    # Exclude validation indices from training dataset
    train_indices = [i for i in range(num_samples) if i not in random_indices]
    train_imgs_filtered = [train_imgs[i] for i in train_indices]
    train_masks_filtered = [train_masks[i] for i in train_indices]

    transform_fn, transform_fn_crop = define_transform_fn(
        patch_size=patch_size)

    train_dataset = Dataset(train_imgs_filtered,
                            train_masks_filtered, transform_fn)
    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=True)
    # no augumentation on valdation set
    val_dataset = Dataset(val_imgs, val_masks, transform_fn_crop)
    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )

    # print("Number of train imgs: " + str(len(train_imgs_filtered)))
    # print("Number of val imgs: " + str(len(val_imgs)))

    return training_loader, validation_loader
```

# Define training



```python
# Code inspired by Jaroslav Knotek
logging.basicConfig()
logger = logging.getLogger("training")
logger.setLevel(logging.DEBUG)


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    epochs=0,
    lr=0,
    device="cpu",
    scheduler=None,
    trial="0",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_model_state = None
    best_val_loss = float("inf")

    if scheduler:
        scheduler.optimizer = optimizer

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

        loss_train, loss_val = run_epoch(train_epoch_fn, eval_epoch_fn)
        train_losses.append(loss_train)
        validation_losses.append(loss_val)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_model_state = copy.deepcopy(model.state_dict())

        trial.report(loss_val, step=epoch)

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
        logger.info(
            f"{epoch=} {loss_val=:.5f}  Learning Rate = {current_lr:.6e}")

    return {"train_loss": train_losses, "val_loss": validation_losses}, best_model_state


def run_epoch(train_epoch_fn, validate_epoch_fn):
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

# Padding

```python
# Code inspired by Jaroslav Knotek
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
    lh, uh = int((new_h - h) // 2), int(new_h - h) - int((new_h - h) // 2)
    lw, uw = int((new_w - w) // 2), int(new_w - w) - int((new_w - w) // 2)
    pads = (lw, uw, lh, uh)

    out = F.pad(x, pads, "replicate")

    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2]: -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0]: -pad[1]]
    return x
```

### Prediction

```python editable=true slideshow={"slide_type": ""}
def predict(img, model, device, probability_threshold, pad_stride=32):
    img_3d = np.stack([img] * 1)
    tensor = torch.from_numpy(img_3d).to(device)[None]
    padded_tensor, pads = pad_to(tensor, pad_stride)
    res_tensor = model(padded_tensor)
    res_unp = unpad(res_tensor, pads)
    # activation function
    res_unp = torch.sigmoid(res_unp)
    res_unp_binary = (res_unp > probability_threshold).float()
    return res_unp_binary.squeeze(0).squeeze(0)
```

## Training and optimalization with OPTUNA

```python
def get_scheduler(optimizer, scheduler_type, params):
    if scheduler_type == "linear":
        return lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=params["end_factor"],
            total_iters=epochs,
        )
    elif scheduler_type == "warmup_cosine":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=params["T_0"], T_mult=params["T_mult"]
        )
```

```python
def configure_scheduler(trial):
    # scheduler_type = "linear"
    scheduler_type = trial.suggest_categorical(
        "scheduler_type", [None, "warmup_cosine", "linear"]
    )
    scheduler_params = {}

    if scheduler_type == "linear":
        scheduler_params["end_factor"] = trial.suggest_categorical(
            "end_factor", [1e-1, 1e-2, 1e-3]
        )
        scheduler_params["T_0"] = 0

    elif scheduler_type == "warmup_cosine":
        scheduler_params["end_factor"] = 0
        scheduler_params["T_mult"] = 1
        scheduler_params["T_0"] = trial.suggest_categorical(
            "T0", [25, 50]
        )
    else:
        scheduler_params["end_factor"] = 0
        scheduler_params["T_0"] = 0

    return scheduler_type, scheduler_params
```

```python
def configure_model(depth, filters):
    decoder_channels = [filters * 2**i for i in range(depth, 0, -1)]

    model = Unet(
        encoder_depth=depth,
        encoder_weights="imagenet",
        in_channels=1,
        decoder_channels=decoder_channels,
    ).to(device)
    return model
```

```python
def save_model(model, depth, patch_size, filters, trial_number):
    os.makedirs(model_path, exist_ok=True)
    model_save_path = f"{model_path}/best_model_{depth}_{patch_size}_{filters}_trial_{trial_number}.pth"
    torch.save(model.state_dict(), model_save_path)
```

```python
loss_fn = IoULoss()


def loss_wrapper(pred, target_dict):
    return loss_fn(pred, target_dict["y"]).mean()


def objective(trial):
    seed = 678
    set_seed(seed)
    # define hyperparameters
    depth = trial.suggest_categorical("depth", [3, 4, 5])
    patch_size = trial.suggest_categorical("patch_size", [128, 256])
    filters = trial.suggest_categorical("filters", [8, 16, 32])

    # depth = 3
    # patch_size = 128
    # filters = 8

    # lr = 1e-3
    lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-4])

    # probability threshold for the binary mask
    probability_threshold = 0.5

    scheduler_type, scheduler_params = configure_scheduler(trial)

    training_loader, validation_loader = loader(patch_size)

    model = configure_model(depth, filters)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, scheduler_type, scheduler_params)

    # Train the model
    best_val_loss = float("inf")
    with logging_redirect_tqdm():
        loss_dict, best_model_state = train(
            model,
            training_loader,
            validation_loader,
            loss_wrapper,
            epochs=epochs,
            lr=lr,
            device=device,
            scheduler=scheduler,
            trial=trial,
        )

    best_val_loss = min(loss_dict["val_loss"])
    model.load_state_dict(best_model_state)

    save_model(model, depth, patch_size, filters, trial.number)

    # Evaluate on test images - first make predictions
    with torch.no_grad():
        preds = [
            predict(img, model, device, probability_threshold) for img in test_imgs
        ]

    iou_values = eval_save(
        depth,
        patch_size,
        filters,
        lr,
        scheduler_type,
        scheduler_params["end_factor"],
        scheduler_params["T_0"],
        best_val_loss,
        preds,
        trial.number,
        probability_threshold,
        test_masks,
        device,
        test_roi,
        eval_results_path,
    )

    plot_predictions(
        depth,
        patch_size,
        filters,
        preds,
        trial.number,
        predictions_path,
        test_imgs,
        test_masks,
        image_test_names,
    )

    plot_loss(depth, patch_size, filters, trial.number, loss_dict, loss_path)
    plot_histogram(iou_values, depth, patch_size,
                   filters, trial.number, histogram_path)

    return best_val_loss
```

```python
if __name__ == "__main__":
    # Grid search
    # search_space = {
    # 'lr': [1e-2, 1e-3, 1e-4],
    # 'scheduler_type': ["linear"],
    # 'end_factor': [1e-2, 1e-3],
    # }
    search_space = {
        'depth': [5],
        'patch_size': [128],
        'filters': [32],

        'lr': [1e-3],
        'end_factor': [1e-3],
        'scheduler_type': ["linear"],

    }

    study = optuna.create_study(
        storage="sqlite:///dbFINAL2.sqlite3",
        sampler=optuna.samplers.GridSampler(search_space),
        study_name="optimization",
        direction="minimize",
    )

    study.optimize(objective, n_trials=1)

    logger.info(
        f"Best value: {study.best_value} (params: {study.best_params})")
```

```python
exit()
```
