import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.segmentation import mark_boundaries

# Default pixel spacing assumption (0.5 mm per pixel)
PIXEL_SPACING = 0.5  # mm per pixel

def determine_lobe(centroid, img_shape):
    """
    Determines the approximate tumor location based on the centroid position.
    Assumes a standard axial view MRI.
    """
    h, w = img_shape[:2]
    x, y = centroid

    # Define hemispheres
    hemisphere = "Left Hemisphere" if x < w / 2 else "Right Hemisphere"

    # Define anterior-posterior regions
    if y < h / 3:
        region = "Frontal Lobe"
    elif y < 2 * h / 3:
        region = "Parietal Lobe"
    else:
        region = "Occipital/Temporal Lobe"

    return f"{region} ({hemisphere})"

def extract_tumor_info(binary_mask, img_shape):
    """
    Extracts tumor size, location, height, width, and determines its brain region.
    """
    binary_mask = (binary_mask > 0.5).astype(np.uint8)  # Ensure binary format

    # Find connected components (tumor regions)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

    # Ignore background (label 0)
    tumor_sizes_px = stats[1:, cv2.CC_STAT_AREA]  # Areas in pixels
    tumor_bboxes = stats[1:, :4]  # Bounding boxes: (x, y, width, height)
    tumor_centroids = centroids[1:]  # Centroids (x, y)

    # Convert tumor size to mm²
    tumor_sizes_mm2 = tumor_sizes_px * (PIXEL_SPACING ** 2)

    # Extract height and width in pixels and mm
    tumor_heights_px = stats[1:, cv2.CC_STAT_HEIGHT]
    tumor_widths_px = stats[1:, cv2.CC_STAT_WIDTH]
    tumor_heights_mm = tumor_heights_px * PIXEL_SPACING
    tumor_widths_mm = tumor_widths_px * PIXEL_SPACING

    # Determine tumor location
    tumor_locations = [determine_lobe(centroid, img_shape) for centroid in tumor_centroids]

    return tumor_sizes_px, tumor_sizes_mm2, tumor_bboxes, tumor_centroids, tumor_heights_mm, tumor_widths_mm, tumor_locations

def analyze_mask_tif(mask_path, original_image_path=None):
    """
    Loads a binary segmentation mask from a .tif file and analyzes tumor size, location, height, width, and region.
    """
    mask = tiff.imread(mask_path)  # Load .tif mask
    mask = (mask > 0.5).astype(np.uint8)  # Convert to binary

    # Load original MRI image if provided
    if original_image_path:
        img = tiff.imread(original_image_path)
    else:
        img = np.zeros_like(mask)  # If no MRI, use black background

    # Extract tumor info
    tumor_sizes_px, tumor_sizes_mm2, tumor_bboxes, tumor_centroids, tumor_heights_mm, tumor_widths_mm, tumor_locations = extract_tumor_info(mask, img.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')

    # Overlay segmentation mask (red)
    img_overlay = mark_boundaries(img, mask, color=(1, 0, 0))
    ax.imshow(img_overlay, alpha=0.6)

    for i, (size_px, size_mm2, (x, y, w, h), centroid, h_mm, w_mm, location) in enumerate(
        zip(tumor_sizes_px, tumor_sizes_mm2, tumor_bboxes, tumor_centroids, tumor_heights_mm, tumor_widths_mm, tumor_locations)):

        # Draw bounding box
        rect = plt.Rectangle((x, y), w, h, edgecolor='red', linewidth=2, fill=False)
        ax.add_patch(rect)

        # Mark centroid
        ax.scatter(centroid[0], centroid[1], color='yellow', marker='x', s=100)

        # Display tumor size, height, width, and location
        ax.text(x, y - 10, f"Size: {size_mm2:.2f} mm²\nH: {h_mm:.2f} mm, W: {w_mm:.2f} mm\n{location}",
                color='red', fontsize=10, weight='bold')

    plt.title("Tumor Analysis from TIFF Mask")
    plt.axis('off')
    plt.show()

    # Print tumor details
    for i, (size_px, size_mm2, h_mm, w_mm, location) in enumerate(zip(tumor_sizes_px, tumor_sizes_mm2, tumor_heights_mm, tumor_widths_mm, tumor_locations)):
        print(f"Tumor {i+1}: {size_px} pixels, {size_mm2:.2f} mm², Height: {h_mm:.2f} mm, Width: {w_mm:.2f} mm, Location: {location}")

# Example usage
mask_tif_path = "tumor_mask.tif"  # Replace with actual .tif mask path
original_mri_tif_path = "mri_image.tif"  # Optional, if available
analyze_mask_tif(mask_tif_path, original_mri_tif_path)
