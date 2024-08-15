import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff


def create_mask_from_coords(coords, shape):
    """Create a binary mask from coordinates."""
    mask = np.zeros(shape, dtype=np.int8)
    coords = np.round(coords).astype(int)
    mask[coords[:, 0], coords[:, 1]] = 1
    return mask


def calculate_confusion_matrix(pred_mask, gt_mask):
    """Calculate confusion matrix elements."""
    TP = np.sum((pred_mask == 1) & (gt_mask == 1))
    TN = np.sum((pred_mask == 0) & (gt_mask == 0))
    FP = np.sum((pred_mask == 1) & (gt_mask == 0))
    FN = np.sum((pred_mask == 0) & (gt_mask == 1))
    return TP, TN, FP, FN


def calculate_dice(TP, FP, FN):
    """Calculate Dice coefficient."""
    return (2 * TP) / (2 * TP + FP + FN)


def calculate_iou(TP, FP, FN):
    """Calculate IoU (Jaccard Index)."""
    return TP / (TP + FP + FN)


def calculate_sensitivity(TP, FN):
    """Calculate Sensitivity."""
    return TP / (TP + FN)


def calculate_fpr(FP, TN):
    """Calculate False Positive Rate."""
    return FP / (FP + TN)


def calculate_rmse(pred_mask, gt_mask):
    """Calculate Root Mean Square Error."""
    mse = np.mean((pred_mask - gt_mask) ** 2)
    return np.sqrt(mse)


def calculate_hausdorff(pred_mask, gt_mask):
    """Calculate Hausdorff Distance."""
    return max(directed_hausdorff(pred_mask, gt_mask)[0],
               directed_hausdorff(gt_mask, pred_mask)[0])


def calculate_all_metrics(pred_coords, gt_coords, image_shape):
    """
    Calculate all metrics for segmentation evaluation.

    Args:
    pred_coords (np.array): Predicted coordinates of shape (N, 2)
    gt_coords (np.array): Ground truth coordinates of shape (M, 2)
    image_shape (tuple): Shape of the original image (height, width)

    Returns:
    dict: Dictionary containing all calculated metrics
    """
    # Create binary masks from coordinates
    pred_mask = create_mask_from_coords(pred_coords, image_shape)
    gt_mask = create_mask_from_coords(gt_coords, image_shape)

    # Calculate confusion matrix elements
    TP, TN, FP, FN = calculate_confusion_matrix(pred_mask, gt_mask)

    # Calculate all metrics
    dice = calculate_dice(TP, FP, FN)
    iou = calculate_iou(TP, FP, FN)
    sensitivity = calculate_sensitivity(TP, FN)
    fpr = calculate_fpr(FP, TN)
    rmse = calculate_rmse(pred_mask, gt_mask)
    hausdorff = calculate_hausdorff(pred_mask, gt_mask)

    return {
        "Dice": dice,
        "IoU": iou,
        "Sensitivity": sensitivity,
        "FPR": fpr,
        "RMSE": rmse,
        "Hausdorff Distance": hausdorff
    }


def get_metrics_df(preds, gts, scans, index_list):
    """
    Calculate metrics for all scans and create a pandas DataFrame.

    Args:
    preds (dict): Dictionary of predicted coordinates for each scan
    gts (dict): Dictionary of ground truth coordinates for each scan
    scans (dict): Dictionary of original scans (needed for shape information)
    index_list (list): List of scan indices or filenames to process

    Returns:
    pd.DataFrame: DataFrame containing metrics for all processed scans
    """
    results = []

    for idx in index_list:
        metrics = calculate_all_metrics(preds[idx], gts[idx], scans[idx].shape)
        metrics['Scan'] = idx  # Add the scan index/filename to the metrics
        results.append(metrics)

    # Create DataFrame from the list of metric dictionaries
    df = pd.DataFrame(results)

    return df






### ALL FUNCTIONS BELOW THIS ARE DEPRECATED USE THE ONES ABOVE
#
# def trace_MSE(edge_pred, edge_true):
#     '''
#     Return the mean squared error between true edge and edge prediction.
#
#     INPUTS:
#     ---------------
#         edge_pred (2darray) : Array storing pixel coordinates (in yx-space) of edge prediction.
#
#         edge_true (2darray) : Ground truth array of edge of interest.
#     '''
#     N = edge_pred.shape[0]
#     if edge_pred.ndim == 1:
#         edge_pred = edge_pred.reshape(-1, 1)
#     return np.round((1 / N) * np.sum((edge_pred[:, 0] - edge_true[:, 0]) ** 2), 4)
#
#
# def trace_relarea(edge_pred, edge_true):
#     '''
#     Return the relative change in area dictated by ground truth edge and edge prediction. This is equivalent to intersection-over-union.
#
#     INPUTS:
#     ---------------
#         edge_pred (2darray) : Array storing pixel coordinates (in yx-space) of edge prediction.
#
#         edge_true (2darray) : Ground truth array of edge of interest.
#     '''
#     N = edge_pred.shape[0]
#     if edge_pred.ndim == 1:
#         edge_pred = edge_pred.reshape(-1, 1)
#     true_area = (np.sum(N - edge_true[:, 0]) / N ** 2)
#     pred_area = (np.sum(N - edge_pred[:, 0]) / N ** 2)
#     return np.round(np.abs((true_area - pred_area) / true_area), 5)
#
#
def calculate_dice2(pred_coords, gt_coords, image_shape):
    """
    Calculate DICE coefficient for two sets of coordinates.

    Args:
    pred_coords (np.array): Predicted coordinates, shape (N, 2)
    gt_coords (np.array): Ground truth coordinates, shape (M, 2)
    image_shape (tuple): Shape of the original image (height, width)

    Returns:
    float: DICE coefficient
    """
    # Create binary masks from coordinates
    pred_mask = np.zeros(image_shape, dtype=bool)
    gt_mask = np.zeros(image_shape, dtype=bool)

    # Fill in the masks
    pred_mask[pred_coords[:, 0], pred_coords[:, 1]] = True
    gt_mask[gt_coords[:, 0], gt_coords[:, 1]] = True

    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    # Calculate DICE coefficient
    dice = (2. * intersection.sum() + 1e-8) / (pred_mask.sum() + gt_mask.sum() + 1e-8)

    return dice


def calculate_iou2(pred_coords, gt_coords, image_shape):
    """
    Calculate IoU (Intersection over Union) for two sets of coordinates.

    Args:
    pred_coords (np.array): Predicted coordinates, shape (N, 2)
    gt_coords (np.array): Ground truth coordinates, shape (M, 2)
    image_shape (tuple): Shape of the original image (height, width)

    Returns:
    float: IoU score
    """
    # Create binary masks from coordinates
    pred_mask = np.zeros(image_shape, dtype=bool)
    gt_mask = np.zeros(image_shape, dtype=bool)

    # Fill in the masks
    pred_mask[pred_coords[:, 0], pred_coords[:, 1]] = True
    gt_mask[gt_coords[:, 0], gt_coords[:, 1]] = True

    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    # Calculate IoU
    iou = intersection.sum() / (union.sum() + 1e-8)

    return iou


# def trace_dicecoef(edge_pred, edge_true, jaccard=False):
#     '''Return the DICE similarity coefficient between the edge prediction and ground truth.
#
#     INPUTS:
#     ---------------
#         edge_pred (2darray) : Array storing pixel coordinates (in yx-space) of edge prediction.
#
#         edge_true (2darray) : Ground truth array of edge of interest.
#
#         jaccard (bool) : If flagged, return Jaccard index instead of DICE (J = D / (2-D) or D = 2J / (J+1))
#    '''
#     N = edge_pred.shape[0]
#     if edge_pred.ndim == 1:
#         edge_pred = edge_pred.reshape(-1, 1)
#     pred_bin = np.zeros((N, N))
#     true_bin = np.zeros_like(pred_bin)
#     for i in range(N):
#         pred_bin[int(edge_pred[i, 0]):, i] = 1
#         true_bin[int(edge_true[i, 0]):, i] = 1
#     jacc = (np.sum(pred_bin * true_bin) / np.sum(np.clip((pred_bin + true_bin), 0, 1)))
#
#     if jaccard:
#         return np.round(jacc, 4)
#     else:
#         return np.round((2 * jacc / (jacc + 1)), 4)
#
