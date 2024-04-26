import torch
import torchvision.transforms as transforms

def evaluate_IoU(ground_truth, inferences, width):
    iou_scores = []

    space_between_lines = width / 10
    for gt_mask, pred_mask in zip(ground_truth, inferences):
        intersection = torch.logical_and(gt_mask, pred_mask).sum().float()
        union = torch.logical_or(gt_mask, pred_mask).sum().float()
        iou = intersection / union if union != 0 else 0

        # IoU for ten lines
        line_iou = []
        for line_number in range(10):
            x = int(space_between_lines / 2 + (line_number * space_between_lines))
            line_gt = gt_mask[:, x]
            line_pred = pred_mask[:, x]

            line_intersection = torch.logical_and(line_gt, line_pred).sum().float()
            line_union = torch.logical_or(line_gt, line_pred).sum().float()
            line_iou.append(line_intersection / line_union if line_union != 0 else 0)

        iou_scores.append(iou.item())


    mean_iou = sum(iou_scores) / len(iou_scores)
    mean_iou_lines = sum (line_iou) / len(line_iou)
    mean_iou_lines = float(mean_iou_lines.item() ) 
    return mean_iou, mean_iou_lines

