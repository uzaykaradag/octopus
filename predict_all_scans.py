import os
import time
import numpy as np
from tqdm import tqdm
import octopus.dataset as ds
from octopus.preprocessing import compute_grad_image
from octopus.core.predict import trace_elm

# Define the output directory
OUTPUT_DIR = '/Users/uzaykaradag/Developer/octopus/predictions/2024-08-13-02'

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the dataset
test_scans, test_initial_elms, ground_truth_elms = ds.load_dataset()

# Start timing
start_time = time.time()

# Process each scan in the test set
for i, (filename, scan) in enumerate(tqdm(test_scans.items(), desc="Processing scans")):
    # Compute gradient image
    grad_img = compute_grad_image(scan)

    # Get initial ELM for this scan
    initial_elm = test_initial_elms[filename]

    # Trace ELM
    predicted_elm = trace_elm(grad_img, initial_elm)

    # Save the result
    output_filename = os.path.join(OUTPUT_DIR, f"{filename}.npy")
    np.save(output_filename, predicted_elm)

    # Update and display progress
    elapsed_time = time.time() - start_time
    avg_time_per_scan = elapsed_time / (i + 1)
    remaining_scans = len(test_scans) - (i + 1)
    estimated_time_left = avg_time_per_scan * remaining_scans

    print(f"\nProcessed {i + 1}/{len(test_scans)} scans")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Estimated time remaining: {estimated_time_left:.2f} seconds")

# Final timing
total_time = time.time() - start_time
print(f"\nAll scans processed. Total time: {total_time:.2f} seconds")