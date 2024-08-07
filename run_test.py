import time
import multiprocessing as mp

from octopus.core import metrics
from octopus.dataset import load_dataset
from octopus.experiments.experiment import trace_elm
from octopus.preprocessing import kernel_builder, compute_grad_image

def compute_gradient(args):
    i, scan, kernel, total_scans = args
    grad = compute_grad_image(scan, kernel)
    print(f"Completed gradient: {i + 1} / {total_scans}")
    return i, grad

def process_scan(args):
    i, grad, base, gt, scan_shape = args
    start = time.time()
    pred = trace_elm(grad, base, num_runs=50)

    dice = metrics.calculate_dice(pred, gt, scan_shape)
    iou = metrics.calculate_iou(pred, gt, scan_shape)
    end = time.time()

    print(f"Completed scan #{i + 1} in {end - start:.2f} seconds")
    print(f"DICE: {dice}")
    print("-------------------")

    return i, pred, dice, iou

if __name__ == '__main__':
    scans, base, gt = load_dataset()
    kernel = kernel_builder([21, 5])

    # Limit to 10 scans
    num_scans = 10
    scans = scans[:num_scans]
    base = base[:num_scans]
    gt = gt[:num_scans]

    # Parallelize gradient computation
    pool = mp.Pool(processes=mp.cpu_count())
    total_scans = len(scans)
    grad_args = [(i, scan, kernel, total_scans) for i, scan in enumerate(scans)]
    grad_results = pool.map(compute_gradient, grad_args)
    pool.close()
    pool.join()

    # Sort and extract gradients
    grad_results.sort(key=lambda x: x[0])
    grads = [result[1] for result in grad_results]

    # Parallelize scan processing
    pool = mp.Pool(processes=mp.cpu_count())
    scan_args = [(i, grads[i], base[i], gt[i], scans[i].shape) for i in range(total_scans)]
    results = pool.map(process_scan, scan_args)
    pool.close()
    pool.join()

    # Sort and extract results
    results.sort(key=lambda x: x[0])
    preds = [result[1] for result in results]
    dice_list = [result[2] for result in results]
    iou_list = [result[3] for result in results]

    # Print summary
    print("\nSummary:")
    print(f"Average DICE: {sum(dice_list) / len(dice_list):.4f}")
    print(f"Average IoU: {sum(iou_list) / len(iou_list):.4f}")