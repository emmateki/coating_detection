import os
import json
import numpy as np

import torch
from sklearn.neighbors import KDTree
from torchmetrics import JaccardIndex, Dice

# Helper Functions
# ================


def _transform(inference):
    """Remove the first and second dimension of the tensor."""
    return inference.squeeze(0).squeeze(0)


def _is_valid_shape(gt, infc):
    """Ensure the ground truth and inference tensors have the same shape."""
    assert gt.shape == infc.shape, "Both tensors must have the same shape."


def convert_to_serializable(obj):
    """Convert objects to a format that can be serialized to JSON."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(data, file_path_JSON):

    # Save results to JSON
    if not os.path.exists(file_path_JSON):
        with open(file_path_JSON, "w") as f:
            json.dump([], f)

    serializable_data = {k: convert_to_serializable(v) for k, v in data.items()}

    # laod existing data
    with open(file_path_JSON, "r") as f:
        existing_data = json.load(f) if os.path.exists(file_path_JSON) else []
    existing_data.append(serializable_data)

    # dump data back to file
    with open(file_path_JSON, "w") as f:
        json.dump(existing_data, f, indent=4)


# Evaluation Functions
# ====================


def evaluate_2d(ground_truths, inferences, metrics, device):
    """Evaluate 2D metrics between ground truth and inferences."""
    gts = [gt.to(device) for gt in ground_truths]
    infs_trans = [_transform(infc).to(device) for infc in inferences]

    for gt, infc in zip(gts, infs_trans):
        _is_valid_shape(gt, infc)

    eval_results = {
        name: [metric(gt, infc) for gt, infc in zip(gts, infs_trans)]
        for name, metric in metrics.items()
    }

    return eval_results


def find_start_end_positions(y_col):
    y_col_cpu = y_col.cpu().numpy()
    max_y = y_col_cpu.shape[0]
    y_found_start = np.argmax(y_col_cpu == 1)
    y_found_end = max_y - 1 - np.argmax(np.flip(y_col_cpu) == 1)
    return y_found_start, y_found_end


def calculate_length_difference(y_found_start, y_found_end, length):
    return abs(abs(y_found_end - y_found_start) - length)


def process_roi_row(roi_row, inference, device):
    x1, x2, y1, y2, length = map(float, roi_row[3:8])
    y_col = inference[:, int(x1)].to(device)

    if not torch.all(y_col == 0):
        y_found_start, y_found_end = find_start_end_positions(y_col)
    else:
        y_found_start, y_found_end = 0, 0

    length_diff = calculate_length_difference(y_found_start, y_found_end, int(length))
    start_diff = abs(y1 - y_found_start) if not np.isnan(y1) else np.nan
    end_diff = abs(y2 - y_found_end) if not np.isnan(y2) else np.nan

    return start_diff, end_diff, length_diff


def evaluate_1d(test_roi, inferences, num_of_lines_per_row, device):
    """Evaluate 1D measurements such as start, end, and length differences."""
    infs_trans = [_transform(infc) for infc in inferences]
    start_diffs, end_diffs, length_diffs = [], [], []

    for num, inference in enumerate(infs_trans):
        for i in range(
            num * num_of_lines_per_row,
            num * num_of_lines_per_row + num_of_lines_per_row,
        ):
            start_diff, end_diff, length_diff = process_roi_row(
                test_roi[i], inference, device
            )
            start_diffs.append(start_diff)
            end_diffs.append(end_diff)
            length_diffs.append(length_diff)

    start_end_diffs = np.concatenate([start_diffs, end_diffs])
    return start_end_diffs, length_diffs


def calculate_mean_border_distance(pred_border_coords, gt_border_coords):
    """Calculate the mean border distance using KDTree for efficiency."""
    tree = KDTree(gt_border_coords)
    distances, _ = tree.query(pred_border_coords, k=1)
    return np.mean(distances)


def find_border_pixels(mask):
    border_mask = np.logical_xor(
        mask, np.pad(mask[1:, :], ((0, 1), (0, 0)), mode="constant")
    )
    border_mask = np.logical_or(
        border_mask,
        np.logical_xor(mask, np.pad(mask[:, 1:], ((0, 0), (0, 1)), mode="constant")),
    )
    return np.column_stack(np.nonzero(border_mask))


def evaluate_border_distance(preds, gt):
    mean_border_distances = []
    for pred, target in zip(preds, gt):
        pred_border_coords = find_border_pixels(pred.cpu().numpy())
        gt_border_coords = find_border_pixels(target.cpu().numpy())
        mean_border_distance = calculate_mean_border_distance(
            pred_border_coords, gt_border_coords
        )
        mean_border_distances.append(mean_border_distance)
    return mean_border_distances


def calculate_fp_fn_percentage(pred, gt):
    """Calculate the false positive and false negative pixels percentages."""
    true_positives = torch.sum((pred == 1) & (gt == 1)).item()
    false_positives = torch.sum((pred == 1) & (gt == 0)).item()
    false_negatives = torch.sum((pred == 0) & (gt == 1)).item()
    total_pixels = gt.shape[0] * gt.shape[1]

    fp_percentage = false_positives / total_pixels
    fn_percentage = false_negatives / total_pixels

    return fp_percentage, fn_percentage


# Evaluation and Saving Function
# ==============================


def eval_save(
    depth,
    patch_size,
    filters,
    lr,
    scheduler_type,
    end_factor,
    T_0,
    loss,
    preds,
    trial,
    probability_threshold,
    test_masks,
    device,
    test_roi,
    file_path_JSON,
):
    """Evaluate the model, save results, and generate histograms."""
    mask_arrays = [np.array(mask) for mask in test_masks]
    dtype = torch.int
    gt = [
        torch.tensor((mask > probability_threshold).astype(np.int32), dtype=dtype)
        for mask in mask_arrays
    ]

    metrics = {"IoU": JaccardIndex(task="binary").to(device)}
    results = evaluate_2d(gt, preds, metrics, device)

    iou_values = [res.cpu().numpy() for res in results["IoU"]]

    # Calculate False Positive and Negative Percentages

    fp_percentages, fn_percentages = calculate_fp_fn_percentage(preds, gt)
    mean_fp_percentage = np.mean(fp_percentages)
    mean_fn_percentage = np.mean(fn_percentages)

    # Evaluate Differences in ROI files
    start_end_diffs, length_diff = evaluate_1d(test_roi, preds, 10, device)
    total_diffs = np.nanmean(start_end_diffs)
    length_diff_total = np.nanmean(length_diff)

    # Mean Border Distance Calculation
    mean_border_distances = evaluate_border_distance(preds, gt)

    data = {
        "Depth": depth,
        "Patch size": patch_size,
        "Filters": filters,
        "Trial": trial,
        "LR": lr,
        "scheduler_type": scheduler_type,
        "end_factor": end_factor,
        "T_0": T_0,
        "Mean IoU": np.mean(iou_values),
        "Min IoU": np.min(iou_values),
        "Mean False Positive %": mean_fp_percentage,
        "Mean False Negative %": mean_fn_percentage,
        "Total diffs in start and end": total_diffs,
        "Length diff": length_diff_total,
        "Mean Border Distance": np.mean(mean_border_distances),
        "Val Loss": loss,
    }

    save_results(data, file_path_JSON)

    return iou_values
