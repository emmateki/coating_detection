---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: denoiseg
    language: python
    name: denoiseg
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
```

# Data Preparation

```python
data_root = pathlib.Path('ML_data')
```

```python
import scipy.ndimage as ndi
from tqdm.auto import tqdm 

def imread(p):
    img = imageio.imread(p)
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm= (img - img_min)/(img_max-img_min)
    return np.float32(img_norm)

def imread_mask(p):
    img = imread(p)
    return np.float32(img == 1) # ensure only two values 1.0 and 0.0
    
def read_set(root,set_name):
    str_set_path = root/f'{set_name}_data'
    x_root = str_set_path/ f"{set_name}_x"
    y_root = str_set_path/f"{set_name}_y"
    
    x_paths = list(x_root.glob("*.png"))
    # ensure the same order
    y_paths = [y_root / p.name for p in x_paths]

    # casting to npfloat
    x = list(tqdm(map(imread,x_paths),total = len(x_paths),desc = "Reading x"))
    
    y = list(tqdm(map(imread_mask,y_paths),total = len(x),desc = "Reading y"))

    
    # HACK : resizing y to have same dimensions as x
    y_resized = []
    for xx,yy in zip(x,y):
        zoom = xx.shape[0]/yy.shape[0],xx.shape[0]/yy.shape[0]
        yy_new = np.float32(ndi.zoom(yy,zoom) == 1)
        y_resized.append(yy_new)
        
    
    return x,y_resized

train_imgs,train_masks = read_set(data_root, 'train')
test_imgs,test_masks = read_set(data_root, 'test')

assert len(train_imgs) == len(train_masks)
assert len(test_imgs) == len(test_masks)
```

```python


for x,y in zip(train_imgs + test_imgs,train_masks + test_masks):
    z = np.zeros_like(y)
    red_mask = np.dstack([y,z,z,y])*255
    
    plt.imshow(x)
    plt.imshow(red_mask)
    plt.show()
```

# Data Alteration

It can be seen that majority of the image is homogenous. That will make the detection harder (with as few images as we have available)

Image will be cropped to the parts where the oxide is located

# TODO

... also, crop sides where the oxides is but the label is not available.

Notice that the patch size is defined here. It's because we need to discuss if it is the right value based on the data. 


# Augumentation

Uses albumentation.

```python
import albumentations as A
def setup_augumentation(
    patch_size,
    elastic=False,  # True
    brightness_contrast=False,
    flip_vertical=False,
    flip_horizontal=False,
    blur_sharp_power=None,  # 1
    noise_val=None,  # .01
    rotate_deg=None,  # 90
    interpolation=2, # constant representing cv2.INTER_CUBIC
):
    patch_size_padded = int(patch_size * 1.5)
    transform_list = [
        A.PadIfNeeded(patch_size_padded, patch_size_padded),
        A.RandomCrop(patch_size_padded, patch_size_padded),
    ]

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
patch_size = 128 # or more?

transform_fn = setup_augumentation(
    patch_size,
    elastic = True,
    brightness_contrast=True,
    flip_vertical=False, # Oxides never flip vertically
    flip_horizontal=True,
    blur_sharp_power=None, # hopefully microscopes don't differ in blur much
    noise_val=.01,
    rotate_deg=10,
)

for x,y in zip(train_imgs + test_imgs,train_masks + test_masks):
    x=np.float32(x)
    transformed = transform_fn(image=x, mask=y)
    tr_x = transformed["image"]
    tr_y = transformed["mask"]

    _,axs = plt.subplots(1,2)
    axs[0].imshow(tr_x,vmin=0,vmax=1)
    axs[1].imshow(tr_y)

```

# Notice 
... that even human would have troubles to see what is the oxide and what is not. Maybe we should crop the image and increase the patch size


# Training

Inspired by tutorial https://pytorch.org/tutorials/beginner/introyt/trainingyt.html


## Dataset

```python
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
        
        transformed = transform_fn(image=x, mask=y)
        tr_x = transformed["image"]
        tr_y = transformed["mask"]
    
        return {
            "x": tr_x[None],
            "y": tr_y[None],
        }
```

### Train/Val Split

This is important to prevent overfitting

```python
batch_size = 32
train_val_split = .2

def ensure_at_least_batch(data,batch_size):
    return (train_imgs*batch_size)[:batch_size]

val_size = int(len(train_imgs) * train_val_split)
train_size = len(train_imgs) - val_size

just_train_imgs = train_imgs[:train_size]
just_train_masks = train_masks[:train_size]

val_imgs = train_imgs[-val_size:]
val_imgs_res = ensure_at_least_batch(val_imgs,batch_size)
```

### Resize

To have at least one batch

```python

train_imgs_res = ensure_at_least_batch( just_train_imgs,batch_size)
train_masks_res = ensure_at_least_batch(just_train_masks,batch_size)

train_ds = Dataset(
    train_imgs_res,
    train_masks_res,
    transform_fn
)
training_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

val_masks = train_masks[-val_size:]
val_masks_res = ensure_at_least_batch(val_masks,batch_size)

# There is no augumentation applied on val!
transform_fn_val  = setup_augumentation(patch_size)
val_ds = Dataset(
    train_imgs_res,
    train_masks_res,
    transform_fn_val
)
validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
```

## Model

See unet.py for more detail.

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
        epochs = 100
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
with torch.no_grad():
    tensors =[torch.Tensor(img[None][None]) for img in test_imgs]
    preds = [model(t).cpu().detach()[0][0] for t in tensors]

for imgs in zip(test_imgs,test_masks, preds):
    fig,axs = plt.subplots(1,3,figsize=(21,7))
    _=[ax.imshow(img) for ax,img in zip(axs,imgs)]
    plt.show()
```

# EXPERMINETS

- More epochs (looks like it's still learning something)
  - 100 epochs is 2 mins on my PC.
  - It will be seconds on GPU
- Better data preparation?
  - Crop sides where valid regions are not labeled
  - Remove bottom where "nothing happens"
- Play around with different `patch_size`
- Different augumentation?
- More data?

- At any moment, think about how many different experiments you have to do.
- Every experiment should have it's parametrization, results and evaluation stored
  - It's likely you will need it later
    

