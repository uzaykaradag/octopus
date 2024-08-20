import numpy as np
import random
from octopus.core.predict import trace_elm
from octopus.core.metrics import calculate_all_metrics
from octopus.dataset import load_dataset
import os
import time
from octopus.preprocessing.image_processing import compute_grad_image

# Load the dataset
scans, inits, gts = load_dataset()

# Randomly sample 10 scans
sample_size = 10
sample_keys = random.sample(list(scans.keys()), sample_size)

# Define parameter ranges
num_runs_range = [50, 75, 100, 150]
subset_size_range = [50, 100, 150, 200]
kernel_types = ["RBF", "Matern", "ExpSineSquared", "RationalQuadratic"]
sigma_f_range = [0.5, 1.0, 2.0]
length_scale_range = [1.0, 2.0, 5.0]
alpha_range = [0.1, 1.0, 2.0]  # for RationalQuadratic
nu_range = [0.5, 1.5, 2.5]  # for Matern
period_range = [1.0, 2.0, 5.0]  # for ExpSineSquared

# Create directory for saving results
os.makedirs("optimization_results", exist_ok=True)

# Function to estimate remaining time
def estimate_remaining_time(elapsed_time, current_iteration, total_iterations):
    avg_time_per_iteration = elapsed_time / current_iteration
    remaining_iterations = total_iterations - current_iteration
    estimated_remaining_time = avg_time_per_iteration * remaining_iterations
    return estimated_remaining_time

# Function to format time
def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Function to run optimization for a specific parameter
def optimize_parameter(param_name, param_range, fixed_params, kernel_options):
    total_iterations = len(param_range) * len(sample_keys)
    current_iteration = 0
    start_time = time.time()
    best_dice_score = -1
    best_param_value = None

    for param_value in param_range:
        if param_name in fixed_params:
            fixed_params[param_name] = param_value
        else:
            kernel_options[param_name] = param_value

        print(f"Testing {param_name} = {param_value}")
        print("Fixed params:", fixed_params)
        print("Kernel options:", kernel_options)

        all_metrics = []
        for key in sample_keys:
            current_iteration += 1
            grad = compute_grad_image(scans[key])
            prediction = trace_elm(
                grad, inits[key], kernel_options=kernel_options, **fixed_params
            )

            # Save prediction
            np.save(
                f"optimization_results/{key}_{param_name}_{param_value}.npy", prediction
            )

            # Calculate metrics
            metrics = calculate_all_metrics(prediction, gts[key], scans[key].shape)
            all_metrics.append(metrics)

            # Calculate and print time information
            elapsed_time = time.time() - start_time
            estimated_remaining = estimate_remaining_time(
                elapsed_time, current_iteration, total_iterations
            )
            print(f"Progress: {current_iteration}/{total_iterations}")
            print(f"Time elapsed: {format_time(elapsed_time)}")
            print(f"Estimated time remaining: {format_time(estimated_remaining)}")

        # Average metrics across all sampled scans
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

        print(f"Average metrics for {param_name} = {param_value}:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value}")
        print("\n")

        # Update best dice score and parameter value
        if avg_metrics['Dice'] > best_dice_score:
            best_dice_score = avg_metrics['Dice']
            best_param_value = param_value

    return best_param_value

# Main optimization loop
overall_start_time = time.time()

# Start with optimizing kernel type
fixed_params = {"num_runs": 100, "subset_size": 100}
base_kernel_options = {"sigma_f": 1.0, "length_scale": 2.0}

print("Optimizing kernel type")
best_kernel = optimize_parameter("kernel", kernel_types, fixed_params, base_kernel_options)
base_kernel_options["kernel"] = best_kernel
print(f"Best kernel type: {best_kernel}")

# Now optimize kernel parameters
if best_kernel == "RationalQuadratic":
    base_kernel_options["alpha"] = 1.0
elif best_kernel == "Matern":
    base_kernel_options["nu"] = 1.5
elif best_kernel == "ExpSineSquared":
    base_kernel_options["period"] = 2.0

print("Optimizing sigma_f")
best_sigma_f = optimize_parameter("sigma_f", sigma_f_range, fixed_params, base_kernel_options)
base_kernel_options["sigma_f"] = best_sigma_f
print(f"Best sigma_f: {best_sigma_f}")

print("Optimizing length_scale")
best_length_scale = optimize_parameter("length_scale", length_scale_range, fixed_params, base_kernel_options)
base_kernel_options["length_scale"] = best_length_scale
print(f"Best length_scale: {best_length_scale}")

if best_kernel == "RationalQuadratic":
    print("Optimizing alpha")
    best_alpha = optimize_parameter("alpha", alpha_range, fixed_params, base_kernel_options)
    base_kernel_options["alpha"] = best_alpha
    print(f"Best alpha: {best_alpha}")
elif best_kernel == "Matern":
    print("Optimizing nu")
    best_nu = optimize_parameter("nu", nu_range, fixed_params, base_kernel_options)
    base_kernel_options["nu"] = best_nu
    print(f"Best nu: {best_nu}")
elif best_kernel == "ExpSineSquared":
    print("Optimizing period")
    best_period = optimize_parameter("period", period_range, fixed_params, base_kernel_options)
    base_kernel_options["period"] = best_period
    print(f"Best period: {best_period}")

# Finally, optimize trace_elm parameters
print("Optimizing num_runs")
best_num_runs = optimize_parameter("num_runs", num_runs_range, fixed_params, base_kernel_options)
fixed_params["num_runs"] = best_num_runs
print(f"Best num_runs: {best_num_runs}")

print("Optimizing subset_size")
best_subset_size = optimize_parameter("subset_size", subset_size_range, fixed_params, base_kernel_options)
fixed_params["subset_size"] = best_subset_size
print(f"Best subset_size: {best_subset_size}")

print("Optimization complete. Results saved in 'optimization_results' directory.")
print(f"Total time taken: {format_time(time.time() - overall_start_time)}")

print("\nBest hyperparameters:")
print(f"Kernel type: {best_kernel}")
print(f"Sigma_f: {best_sigma_f}")
print(f"Length scale: {best_length_scale}")
if best_kernel == "RationalQuadratic":
    print(f"Alpha: {best_alpha}")
elif best_kernel == "Matern":
    print(f"Nu: {best_nu}")
elif best_kernel == "ExpSineSquared":
    print(f"Period: {best_period}")
print(f"Num runs: {best_num_runs}")
print(f"Subset size: {best_subset_size}")