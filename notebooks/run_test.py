import multiprocessing as mp
import time

from octopus.core import metrics
from octopus.dataset import load_dataset
from octopus.core.predict import trace_elm
from octopus.preprocessing import kernel_builder, compute_grad_image


def process_scan(i, scan, base, gt, kernel):
    grad = compute_grad_image(scan, kernel)
    print(f"Completed gradient: {i + 1}")

    start = time.time()
    pred = trace_elm(grad, base, num_runs=50)

    dice = metrics.calculate_dice(pred, gt, scan.shape)
    iou = metrics.calculate_iou(pred, gt, scan.shape)
    end = time.time()

    print(f"Completed scan #{i + 1} in {end - start:.2f} seconds")
    print(f"DICE: {dice}")
    print(f"SegNet DICE: {metrics.calculate_dice(base, gt, scan.shape)}")
    print("-------------------")

    return i, grad, pred, dice, iou



if __name__ == '__main__':
    scans, base, gt = load_dataset()
    kernel = kernel_builder([21, 5])

    # Create a pool of workers
    pool = mp.Pool(processes=mp.cpu_count())

    # Submit tasks to the pool
    results = []
    for i, scan in enumerate(scans):
        results.append(pool.apply_async(process_scan, (i, scan, base[i], gt[i], kernel)))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Collect results
    grads = [None] * len(scans)
    preds = [None] * len(scans)
    dice_list = [None] * len(scans)
    iou_list = [None] * len(scans)

    for result in results:
        i, grad, pred, dice, iou = result.get()
        grads[i] = grad
        preds[i] = pred
        dice_list[i] = dice
        iou_list[i] = iou


