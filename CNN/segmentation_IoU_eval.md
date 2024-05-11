---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: torch_cv
    language: python
    name: torch_cv
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
from torchmetrics import IoU

device = 'cuda' # if cuda available
```

# Data Preparation

```python
data_root = pathlib.Path('/home/knotek/jupyterlab/data/Data_povlak_mech/')
```

```python
import scipy.ndimage as ndi
from tqdm.auto import tqdm 

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
    y_root = str_set_path/f"{set_name}_y"
    
    x_paths = list(x_root.glob("*.png"))
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
        
    return x,y_resized

test_imgs,test_masks = read_set(data_root, 'test')
train_imgs,train_masks = read_set(data_root, 'train')

assert len(train_imgs) == len(train_masks)
assert len(test_imgs) == len(test_masks)
```

# Data Alteration

It can be seen that majority of the image is homogenous. That will make the detection harder (with as few images as we have available)

Image will be cropped to the parts where the oxide is located

# TODO

... also, crop sides where the oxides is but the label is not available.

Notice that the patch size is defined here. It's because we need to discuss if it is the right value based on the data. 


# Croping Images

```python

```

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
    plt.show()
    break # show jusst one

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

### Adjust Dataset Size

... to have at least one batch

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

def predict(img, model, device, pad_stride=32):
    img_3d = np.stack([img] * 1)
    tensor = torch.from_numpy(img_3d).to(device)[None]
    padded_tensor, pads = pad_to(tensor, pad_stride)
    res_tensor = model(padded_tensor)
    res_unp = unpad(res_tensor, pads)
    # convert to binary mask with this threshold 
    res_unp_binary = (res_unp > 0.5).float()  
    return res_unp_binary

with torch.no_grad():
    predictions = [predict(img, model, device) for img in test_imgs]

# test_masks to binary tensors as well
mask_arrays = [np.array(mask) for mask in test_masks]
dtype = torch.float32
mask_tensors = [torch.tensor((mask > 0.5).astype(np.float32), dtype=dtype) for mask in mask_arrays]

metrics = {'IoU': IoU(num_classes=2)}
results = evaluate_2d(mask_tensors, predictions, metrics, device)

print("Mean IoU:", np.mean(results['IoU']))

```

# EXPERMINETS

- More epochs (looks like it's still learning something)
  - 100 epochs is 2 mins on my PC.
  - It will be seconds on GPU
- Better data preparation?
  - Crop sides where valid regions are not labeled
  - Remove bottom where "nothing happens"
- Play around with different `patch_size`
    - 256 is too big
- Different augumentation?
- More data?

- At any moment, think about how many different experiments you have to do.
- Every experiment should have it's parametrization, results and evaluation stored
  - It's likely you will need it later
    


```python
exit ()
```
