import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from octopus.core.metrics import *


def display_scan(scan, predictions, gt_elm=None, credible_intervals=None, figsize=(10, 10)):
    """
    Display the ELM coordinates overlaid on the OCT scan using scatter plots.

    Parameters:
    - scan: 2D numpy array representing the OCT scan
    - 2024-08-13-01: dict where keys are model names and values are numpy arrays of shape (N, 2)
    - gt_elm: numpy array of shape (N, 2) for ground-truth ELM coordinates (optional)
    - credible_intervals: dict with same keys as 2024-08-13-01, values are tuples of lower and upper bounds (optional)
    - figsize: tuple, size of the figure (width, height) in inches
    """

    fig = plt.figure(figsize=figsize)
    plt.suptitle("")

    # Display the scan
    plt.imshow(scan, cmap='gray')

    # Color cycle for different 2024-08-13-01
    colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions)))
    legend_elements = []

    # Plot true ELM if provided
    if gt_elm is not None:
        plt.scatter(gt_elm[:, 1], gt_elm[:, 0], color='green', s=1, alpha=0.5, label='True ELM')
        legend_elements.append(
            Line2D([0], [0], color='green', marker='o', linestyle='', markersize=5, label='True ELM'))

    # Plot 2024-08-13-01 and calculate metrics
    for (model_name, elm_coords), color in zip(predictions.items(), colors):
        plt.scatter(elm_coords[:, 1], elm_coords[:, 0], color=color, s=1, alpha=0.5, label=model_name)

        if gt_elm is not None:
            dice_coef = calculate_dice(elm_coords, gt_elm, image_shape=scan.shape)
            iou_score = calculate_iou(elm_coords, gt_elm, image_shape=scan.shape)
            label = f'{model_name} (DICE: {dice_coef:.4f}, IoU: {iou_score:.4f})'
        else:
            label = model_name

        legend_elements.append(Line2D([0], [0], color=color, marker='o', linestyle='', markersize=5, label=label))

        # Plot credible intervals if provided
        if credible_intervals and model_name in credible_intervals:
            lower, upper = credible_intervals[model_name]
            plt.fill_between(np.arange(len(lower)), lower, upper, color=color, alpha=0.2)

    # Customize the plot
    plt.legend(handles=legend_elements, loc='best', fontsize=10)

    # Remove ticks for a cleaner look
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    return fig


def advanced_display_scan(scan, predictions, gt_elm, figsize=(10, 10), zoom_region=None):
    """
    Display a detailed comparison of ELM line prediction for a single method.
    Parameters:
    - scan: 2D numpy array representing the OCT scan
    - predictions: dict with a single key-value pair where key is the model name and value is numpy array of shape (N, 2)
    - gt_elm: numpy array of shape (N, 2) for ground-truth ELM coordinates
    - figsize: tuple, size of the figure (width, height) in inches
    - zoom_region: tuple (x1, x2, y1, y2) specifying the region to zoom in
    Returns:
    - fig: matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    model_name, pred_coords = next(iter(predictions.items()))

    # Create masks for ground truth and prediction
    shape = scan.shape
    gt_mask = np.zeros(shape, dtype=np.uint8)
    pred_mask = np.zeros(shape, dtype=np.uint8)
    for coord in gt_elm:
        gt_mask[int(coord[0]), int(coord[1])] = 1
    for coord in pred_coords:
        pred_mask[int(coord[0]), int(coord[1])] = 1

    # Compute TP, FP, FN
    TP = cv2.bitwise_and(pred_mask, gt_mask)
    FP = cv2.bitwise_and(pred_mask, cv2.bitwise_not(gt_mask))
    FN = cv2.bitwise_and(cv2.bitwise_not(pred_mask), gt_mask)

    # Create color masks
    colors = np.array([[0, 255, 0], [255, 0, 0], [0, 0, 255]], dtype=np.uint8)  # Green (TP), Red (FP), Blue (FN)

    color_mask = np.zeros((*shape, 3), dtype=np.uint8)
    color_mask[TP == 1] = colors[0]
    color_mask[FP == 1] = colors[1]
    color_mask[FN == 1] = colors[2]

    # Normalize scan to 0-255 range
    scan_normalized = ((scan - scan.min()) / (scan.max() - scan.min()) * 255).astype(np.uint8)
    scan_rgb = cv2.cvtColor(scan_normalized, cv2.COLOR_GRAY2RGB)

    # Blend images
    blended = cv2.addWeighted(scan_rgb, 0.7, color_mask, 0.3, 0)

    # Display the blended image
    ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    ax.set_title("")

    # Remove axis numbering
    ax.set_xticks([])
    ax.set_yticks([])

    # Add zoomed inset if zoom_region is specified
    if zoom_region:
        x1, x2, y1, y2 = zoom_region
        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47],
                              xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
        axins.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axins.set_xticks([])
        axins.set_yticks([])

        # Connect the zoomed region to the inset axes
        ax.indicate_inset_zoom(axins, edgecolor="white")

    # Add color legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.5, label="True Positive"),
        plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.5, label="False Positive"),
        plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5, label="False Negative")
    ]
    ax.legend(handles=legend_elements, loc='lower right', title="Color Legend", fontsize=12)

    # Adjust the layout
    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05, hspace=0, wspace=0)

    return fig

# def plot_elm(scan, elm_coords, figsize=(10, 10), compare=False, true_elm_coords=None):
#     """
#     !DEPRECATED
#     Display the ELM coordinates overlaid on the OCT scan.
#
#     Parameters:
#     - scan: 2D numpy array representing the OCT scan
#     - elm_coords: numpy array of shape (N, 2) representing the predicted ELM coordinates
#     - figsize: tuple, size of the figure (width, height) in inches
#     - compare: boolean, if True, compare predicted ELM coordinates with true ELM coordinates
#     - true_elm_coords: numpy array of shape (N, 2) representing the true ELM coordinates (required if compare is True)
#     """
#
#     print('This function is deprecated use: octopus.plotting.display_scan instead')
#
#     plt.figure(figsize=figsize)
#
#     # Display the scan
#     plt.imshow(scan, cmap='gray')
#
#     label_elm = 'ELM'
#
#     if compare:
#         if true_elm_coords is None:
#             raise ValueError("true_elm_coords must be provided when compare is True")
#
#         label_elm = 'Predicted ELM'
#
#         # Overlay the true ELM coordinates
#         plt.scatter(true_elm_coords[:, 1], true_elm_coords[:, 0], c='green', s=1, alpha=0.5, label='True ELM')
#
#         # Calculate the DICE coefficient
#         dice_coef = trace_dicecoef(elm_coords, true_elm_coords)
#         plt.title(f'OCT Scan with ELM Coordinates (DICE: {dice_coef:.4f})')
#     else:
#         plt.title('OCT Scan with ELM Coordinates')
#
#     # Overlay the predicted ELM coordinates
#     plt.scatter(elm_coords[:, 1], elm_coords[:, 0], c='red', s=1, alpha=0.5, label=label_elm)
#
#     plt.legend()
#     plt.axis()
#     plt.tight_layout()
#     plt.show()

# def plot_results(edge_trace, true_edge, test_img, grad_img, credint=None, string='True Edge vs. Edge Pred'):
#     '''
#     Plot results from edge trace onto test_img, grad_img.
#
#     INPUTS:
#     ---------------
#         edge_pred (2darray) : Array storing pixel coordinates (in yx-space) of edge prediction.
#
#         edge_true (2darray) : Ground truth array of edge of interest.
#
#         test_img (2darray) : Test image.
#
#         grad_img (2darray) : Image gradient.
#
#         credint (bool, default None) : Object to store 95% credible interval of edge prediction.
#
#         string (str, default 'True Edge vs. Edge Pred') : String to display on axis 1.
#     '''
#     # Compute metrics
#     if edge_trace.ndim == 1:
#         edge_trace = edge_trace.reshape(-1, 1)
#     rel_area_diff = trace_relarea(edge_trace, true_edge)
#     dice_coeff = calculate_dice(edge_trace, true_edge, test_img.shape)
#     mse = trace_MSE(edge_trace, true_edge)
#
#     # Plot results
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
#     ax1.imshow(test_img, cmap='gray')
#     ax1.set_title(f'{string}', fontsize=20)
#     ax2.imshow(grad_img, cmap='gray')
#     ax2.set_title(f'MSE: {mse}, Rel. Area Diff: {rel_area_diff}, DICE: {dice_coeff}', fontsize=20)
#     ax1.plot(true_edge[[0, -1], 1], true_edge[[0, -1], 0], 'o', c='r', markersize=5, label='Edge Endpoints')
#     ax2.plot(true_edge[[0, -1], 1], true_edge[[0, -1], 0], 'o', c='r', markersize=5, label='Edge Endpoints')
#     ax1.plot(true_edge[:, 1], edge_trace[:, 0], 'r-', zorder=2, label='Proposed')
#     ax1.plot(true_edge[:, 1], true_edge[:, 0], 'b--', linewidth=2, label='Ground Truth')
#     ax2.plot(true_edge[:, 1], edge_trace[:, 0], 'r-', zorder=2, label='Proposed')
#     ax2.plot(true_edge[:, 1], true_edge[:, 0], 'b--', linewidth=2, label='Ground Truth')
#     if credint is not None:
#         ax1.fill_between(true_edge[:, 1], credint[0], credint[1], alpha=0.5,
#                          color='m', zorder=1, label='95% Credible Region')
#         ax2.fill_between(true_edge[:, 1], credint[0], credint[1], alpha=0.5,
#                          color='m', zorder=1, label='95% Credible Region')
#     ax1_legend = ax1.legend(ax1.get_legend_handles_labels()[1], fontsize=13, ncol=2,
#                             loc='lower right', edgecolor=(0, 0, 0, 1.))
#     ax1_legend.get_frame().set_alpha(None)
#     ax1_legend.get_frame().set_facecolor((1, 1, 1, 1))
#     ax2_legend = ax2.legend(ax2.get_legend_handles_labels()[1], fontsize=13, ncol=2,
#                             loc='lower right', edgecolor=(0, 0, 0, 1.))
#     ax2_legend.get_frame().set_alpha(None)
#     ax2_legend.get_frame().set_facecolor((1, 1, 1, 1))
#     fig.tight_layout()
#
#     return fig
