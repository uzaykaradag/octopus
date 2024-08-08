import multiprocessing as mp

import numpy as np

from octopus.core.edge_tracer import GPEdgeTracer
from octopus.preprocessing import kernel_builder, compute_grad_image


def create_subsets(segnet_elm, num_subsets=3, subset_size=100):
    # Sort the array by x-values (second column)
    sorted_indices = np.argsort(segnet_elm[:, 1])
    sorted_segnet_elm = segnet_elm[sorted_indices]

    # Get unique x-values and their indices
    unique_x, unique_indices = np.unique(sorted_segnet_elm[:, 1], return_index=True)

    # If we have fewer unique x-values than subset_size, we'll use all available
    if len(unique_x) < subset_size:
        subset_size = len(unique_x)

    # Randomly select subset_size unique x-values
    selected_x_indices = np.sort(np.random.choice(len(unique_x), size=subset_size, replace=False))
    selected_x = unique_x[selected_x_indices]

    # Create subsets
    subsets = []
    for _ in range(num_subsets):
        subset = []
        for x in selected_x:
            # Find all points with this x-value
            x_points = sorted_segnet_elm[sorted_segnet_elm[:, 1] == x]
            # Randomly select one point
            selected_point = x_points[np.random.randint(len(x_points))]
            subset.append(selected_point)

        subset = np.array(subset)
        subset[:, [0, 1]] = subset[:, [1, 0]]  # yx to xy
        subsets.append(subset)

    return subsets


def find_discontinuities(coords, jump_threshold):
    sorted_coords = coords[coords[:, 1].argsort()]
    x_diffs = np.diff(sorted_coords[:, 1])
    jump_indices = np.where(x_diffs > jump_threshold)[0]
    return jump_indices, sorted_coords


def filter_discontinuities(coords, jump_indices, sorted_coords, buffer=1):
    mask = np.ones(coords.shape[0], dtype=bool)
    for idx in jump_indices:
        x_start = sorted_coords[idx, 1] - buffer
        x_end = sorted_coords[idx + 1, 1] + buffer
        mask &= ~((coords[:, 1] > x_start) & (coords[:, 1] < x_end))
    filtered_coords = coords[mask]
    return filtered_coords


def trace_elm(edge_map, initial_elm, kernel_options=None, num_runs=3, subset_size=100, discontinuity_threshold=50,
                 buffer=1):
    """Trace the External Limiting Membrane (ELM) using Gaussian Process Edge Tracing."""
    if kernel_options is None:
        kernel_options = {
                'kernel': 'RationalQuadratic',
                'sigma_f': 1.0,
                'length_scale': 2.5,
                'alpha': 2.0
        }
        
    gp_params = {
            'kernel_options': kernel_options,
            'delta_x': 8,
            'score_thresh': 0.5,
            'n_samples': 1000,
            'seed': 1,
            'noise_y': 0.5,
            'keep_ratio': 0.1,
            'pixel_thresh': 5,
            'fix_endpoints': True,
            'return_std': True
        }

    init_points = initial_elm[[0, -1], :][:, [1, 0]]
    coordinate_subsets = create_subsets(initial_elm, subset_size=subset_size, num_subsets=num_runs)

    predictions = []
    for subset in coordinate_subsets:
        gp_params['obs'] = subset
        elm_tracer = GPEdgeTracer(init_points, edge_map, **gp_params)
        prediction, _ = elm_tracer()
        predictions.append(prediction)

    combined_predictions = np.vstack(predictions)
    unique_predictions = np.unique(combined_predictions, axis=0)

    # Remove discontinuities
    discontinuity_indices, sorted_coords = find_discontinuities(initial_elm, discontinuity_threshold)
    final_prediction = filter_discontinuities(unique_predictions, discontinuity_indices, sorted_coords, buffer)

    return final_prediction