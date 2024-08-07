import numpy as np
from octopus.core.edge_tracer import GPEdgeTracer
from octopus.preprocessing import image_processing
from octopus.visualization import plotting
from octopus.core import metrics


def create_subsets(coordinates, num_subsets=3):
    """Create subsets of coordinates for multiple runs."""
    return [coordinates[i::num_subsets] for i in range(num_subsets)]


def find_discontinuities(coordinates, discontinuity_threshold):
    """Find indices where there are large discontinuities in the x-coordinates."""
    sorted_coords = coordinates[coordinates[:, 1].argsort()]
    x_diffs = np.diff(sorted_coords[:, 1])
    discontinuity_indices = np.where(x_diffs > discontinuity_threshold)[0]
    return discontinuity_indices, sorted_coords


def filter_discontinuities(coordinates, discontinuity_indices, sorted_coords, buffer=1):
    """Remove coordinate ranges around discontinuities."""
    mask = np.ones(coordinates.shape[0], dtype=bool)
    for idx in discontinuity_indices:
        x_start = sorted_coords[idx, 1] - buffer
        x_end = sorted_coords[idx + 1, 1] + buffer
        mask &= ~((coordinates[:, 1] > x_start) & (coordinates[:, 1] < x_end))
    return coordinates[mask]


def trace_elm(edge_map, initial_elm, gp_params=None, num_runs=3, discontinuity_threshold=50, buffer=1):
    """Trace the External Limiting Membrane (ELM) using Gaussian Process Edge Tracing."""
    if gp_params is None:
        gp_params = {
            'kernel_options': {
                'kernel': 'RationalQuadratic',
                'sigma_f': 1.0,
                'length_scale': 1.0,
                'alpha': 2.0
            },
            'delta_x': 8,
            'score_thresh': 0.5,
            'N_samples': 1000,
            'seed': 1,
            'noise_y': 0.5,
            'keep_ratio': 0.1,
            'pixel_thresh': 5,
            'fix_endpoints': True,
            'return_std': True
        }

    init_points = initial_elm[[0, -1], :][:, [1, 0]]
    coordinate_subsets = create_subsets(initial_elm, num_subsets=num_runs)

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