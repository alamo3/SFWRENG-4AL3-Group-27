import os
import cv2  # OpenCV for reading 16-bit PNGs
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

def collect_depth_values(root_dir):
  
    all_depth_values = []
    root_path = Path(root_dir)
    
    if not root_path.is_dir():
        print(f"Error: Path '{root_dir}' is not a valid directory.", file=sys.stderr)
        return np.array([])

    print(f"Starting scan in: {root_path.resolve()}")
    
    # Use rglob to recursively find all .png files
    png_files = list(root_path.rglob("*.png"))

    # We will only analyze 1200 files to keep processing time reasonable

    png_files = png_files[:min(1200, len(png_files))]
    
    if not png_files:
        print("No .png files found in the directory tree.")
        return np.array([])

    print(f"Found {len(png_files)} .png files. Analyzing...")
    
    processed_count = 0
    for file_path in tqdm(png_files):
        try:
            # Read the image "as is" (unchanged) to preserve 16-bit depth
            # cv2.IMREAD_UNCHANGED includes the alpha channel if present,
            # but more importantly, it reads the original bit depth.
            image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"Warning: Could not read file (may be corrupt): {file_path}")
                continue
                
            # Check if the image is 16-bit (unsigned 16-bit integer)
            if image.dtype == np.uint16:
                # Flatten the 2D (or 3D if alpha) array into a 1D list of values
                all_depth_values.extend(image.flatten())
                processed_count += 1
            else:
                print(f"Skipping (not 16-bit): {file_path} (dtype: {image.dtype})")
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}", file=sys.stderr)

    print(f"Successfully processed {processed_count} 16-bit depth maps.")
    
    # Convert the list of all values into a single NumPy array
    return np.array(all_depth_values, dtype=np.uint16)

def plot_depth_distribution(depth_values, bins=256):
    """
    Plots a histogram of the given depth values.

    Args:
        depth_values (np.ndarray): Array of depth values.
        bins (int): The number of bins to use for the histogram.
    """
    if depth_values.size == 0:
        print("No depth data to plot.")
        return

    print(f"Plotting histogram for {depth_values.size} data points...")

    plt.figure(figsize=(12, 7))
    
    # Create the histogram
    # We use the full 16-bit range [0, 65535] for the x-axis
    plt.hist(depth_values, bins=bins, range=(0, 65535), color='c', edgecolor='k', alpha=0.7)
    
    plt.title('Distribution of 16-bit Depth Values', fontsize=16)
    plt.xlabel('Depth Value (0 - 65535)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    
    # Set x-axis limits to the full 16-bit range
    plt.xlim(0, 65535)
    
    # Apply a logarithmic scale to the y-axis if the data is
    # heavily skewed (e.g., a massive number of '0' values)
    # This often makes the distribution of non-zero values more visible.
    plt.yscale('log')
    plt.ylabel('Frequency (Count, log scale)')
    print("Note: Y-axis is on a log scale to show detail.")
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def main():
    """
    Main function to parse arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyzes and plots the distribution of depth values "
                    "from 16-bit PNG depth maps in a directory."
    )
    
    parser.add_argument(
        "--directory",
        type=str,
        help="The root directory to scan for 16-bit PNG files."
    )
    
    parser.add_argument(
        "--bins",
        type=int,
        default=256,
        help="Number of bins to use for the histogram. (Default: 256)"
    )
    
    args = parser.parse_args()
    
    all_values = collect_depth_values(args.directory)
    
    if all_values.size > 0:
        plot_depth_distribution(all_values, args.bins)
    else:
        print("No valid 16-bit depth data was found.")

if __name__ == "__main__":
    main()