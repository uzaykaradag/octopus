import os
import numpy as np
from skimage import io, util

# Hardcoded paths
SCAN_DIRECTORY = '/Users/uzaykaradag/Developer/octopus/data/image'
PRED_DIRECTORY = '/Users/uzaykaradag/Developer/octopus/data/pred'
GT_DIRECTORY = '/Users/uzaykaradag/Developer/octopus/data/gt'


def preprocess_scan(scan_path):
    scan = io.imread(scan_path, as_gray=True)
    return util.img_as_float(scan)


def extract_elm_coords(mask_path):
    mask = io.imread(mask_path, as_gray=True)
    binary_mask = mask > 0.5
    elm_coords = np.argwhere(binary_mask)
    return elm_coords[np.argsort(elm_coords[:, 1])]


def get_file_list():
    if not os.path.exists(SCAN_DIRECTORY):
        raise FileNotFoundError(f"Directory not found: {SCAN_DIRECTORY}")

    files = [f for f in os.listdir(SCAN_DIRECTORY) if f.endswith(('.png'))]

    if not files:
        raise ValueError(f"No image files found in {SCAN_DIRECTORY}")

    return files


def load_dataset():
    files = get_file_list()
    scans = dict()
    inits = dict()
    gts = dict()

    for file in files:
        scan_path = os.path.join(SCAN_DIRECTORY, file)
        pred_mask_path = os.path.join(PRED_DIRECTORY, file)
        gt_mask_path = os.path.join(GT_DIRECTORY, file)

        key = file.replace('.png', '')

        scan = preprocess_scan(scan_path)
        scans[key] = scan

        init_elm = extract_elm_coords(pred_mask_path)
        inits[key] = init_elm

        gt_elm = extract_elm_coords(gt_mask_path)
        gts[key] = gt_elm

    return scans, inits, gts
