import numpy as np
from sklearn.model_selection import ParameterSampler, KFold
import octopus.dataset as ds
from octopus.core import predict
from octopus.core.metrics import calculate_dice
import multiprocessing as mp
from functools import partial


def evaluate_kernel_options(scans, inits, gts, kernel_options, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []

    scan_keys = list(scans.keys())

    for train_index, val_index in kf.split(scan_keys):
        val_keys = [scan_keys[i] for i in val_index]

        val_scans = {k: scans[k] for k in val_keys}

        # Train and evaluate
        predictions = {}
        for scan_id, scan in val_scans.items():
            pred = predict.trace_elm(scan, inits[scan_id], kernel_options=kernel_options)
            predictions[scan_id] = pred

        # Calculate Dice coefficient
        dice_scores = [calculate_dice(predictions[k], gts[k], scans[k].shape) for k in val_keys]
        scores.append(np.mean(dice_scores))

    return np.mean(scores)


def evaluate_params(params, scans, inits, gts):
    kernel_params = {
        'kernel': params['kernel'],
        'sigma_f': params['sigma_f'],
        'length_scale': params['length_scale']
    }
    if params['kernel'] == 'RationalQuadratic':
        kernel_params['alpha'] = params['alpha']
    elif params['kernel'] == 'Matern':
        kernel_params['nu'] = params['nu']
    elif params['kernel'] == 'ExpSineSquared':
        kernel_params['period'] = params['period']

    score = evaluate_kernel_options(scans, inits, gts, kernel_params)
    return kernel_params, score


if __name__ == '__main__':
    # Load the dataset
    scans, inits, gts = ds.load_dataset()

    # Calculate initial Dice score
    initial_dice_scores = [calculate_dice(inits[k], gts[k], scans[k].shape) for k in scans.keys()]
    initial_avg_dice = np.mean(initial_dice_scores)
    print(f"Initial average Dice score: {initial_avg_dice:.4f}")

    # Define parameter space
    param_grid = {
        'kernel': ['RBF', 'Matern', 'ExpSineSquared', 'RationalQuadratic'],
        'sigma_f': np.logspace(-1, 2, 5),
        'length_scale': np.logspace(-1, 2, 5),
        'alpha': np.logspace(-1, 2, 5),
        'nu': [0.5, 1.5, 2.5],
        'period': np.logspace(0, 3, 5)
    }

    # Random search
    n_iter = 50  # number of parameter settings that are sampled
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

    # Parallelize the evaluation
    pool = mp.Pool(processes=mp.cpu_count())
    evaluate_params_partial = partial(evaluate_params, scans=scans, inits=inits, gts=gts)
    results = pool.map(evaluate_params_partial, param_list)
    pool.close()
    pool.join()

    # Process results
    for i, (params, score) in enumerate(results):
        print(f"Parameters {i + 1}/{n_iter}:")
        print(f"  {params}")
        print(f"  Dice score: {score:.4f}")

    best_params, best_score = max(results, key=lambda x: x[1])

    print("\nBest parameters:")
    print(f"  {best_params}")
    print(f"Best Dice score: {best_score:.4f}")
    print(f"Improvement over initial: {best_score - initial_avg_dice:.4f}")