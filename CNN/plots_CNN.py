import os
import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_predictions(
    depth,
    patch_size,
    filters,
    preds,
    trial,
    pics_path,
    test_imgs,
    test_masks,
    image_test_names,
):
    fol_name = f"{pics_path}/Pred_{depth}_{patch_size}_{filters}_{trial}"
    os.makedirs(fol_name, exist_ok=True)
    indiv_fol_name = (
        f"{pics_path}/Pred_{depth}_{patch_size}_{filters}_{trial}_individual"
    )
    os.makedirs(indiv_fol_name, exist_ok=True)

    for i, (img, mask, pred) in enumerate(zip(test_imgs, test_masks, preds)):
        fig, axs = plt.subplots(1, 3, figsize=(21, 7))

        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        axs[0].imshow(img)
        axs[1].imshow(mask)
        axs[2].imshow(pred)

        plt.savefig(
            f"{fol_name}/plot_{image_test_names[i]}_{depth}_{patch_size}_{filters}_{trial}.png"
        )
        plt.close(fig)

        plt.imsave(
            f"{indiv_fol_name}/pred_{image_test_names[i]}_{depth}_{patch_size}_{filters}_{trial}.png",
            pred,
        )


def plot_loss(depth, patch_size, filters, trial, loss_dict, loss_path):
    for k, v in loss_dict.items():
        plt.plot(v, label=k)

    best_epoch = np.argmin(loss_dict["val_loss"])
    plt.axvline(best_epoch, label=f"{best_epoch=}")
    plt.title("Loss Visualization")
    plt.legend()
    os.makedirs(f"{loss_path}", exist_ok=True)
    plt.savefig(f"{loss_path}/loss_{depth}_{patch_size}_{filters}_{trial}.png")
    plt.close()

    for k, v in loss_dict.items():
        plt.plot(v, label=k)

    best_epoch = np.argmin(loss_dict["val_loss"])
    plt.axvline(
        best_epoch, label=f"Best Epoch = {best_epoch}", linestyle="--", color="red"
    )
    plt.title("Loss Visualization")
    plt.yscale("log")  # Use a logarithmic scale for the y-axis for better visualization
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.savefig(f"{loss_path}/loss_{depth}_{patch_size}_{filters}_{trial}_LOG.png")
    plt.close()


def plot_histogram(ious, depth, patch_size, filters, trial, histogram_path):
    fol_name = f"{histogram_path}"
    os.makedirs(fol_name, exist_ok=True)

    plt.hist(ious, bins=20, color="blue", alpha=0.7)
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.title("Histogram of IoU values (0 to 1)")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.savefig(
        f"{histogram_path}/histogram_0_to_1_{depth}_{patch_size}_{filters}_{trial}.png"
    )
    plt.close()

    plt.hist(ious, bins=20, color="blue", alpha=0.7)
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.title("Histogram of IoU values")
    plt.grid(True)
    plt.savefig(
        f"{histogram_path}/histogram_{depth}_{patch_size}_{filters}_{trial}.png"
    )
    plt.close()
