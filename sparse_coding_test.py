import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, maximum_filter
import pickle
import zlib

def compute_sparse_motion_guidance(image1, image2, kernel_size=25, grid_stride=80):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    flow_approx = cv2.absdiff(gray2, gray1)

    # Sobel filter for edge detection
    sobelx = cv2.Sobel(flow_approx, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(flow_approx, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.hypot(sobelx, sobely)

    # Topological distance
    edge_binary = (edge_map > np.mean(edge_map)).astype(np.uint8)
    distance_map = distance_transform_edt(1 - edge_binary)

    # Non-maximum suppression
    nms = (distance_map == maximum_filter(distance_map, size=kernel_size))
    keypoints = np.argwhere(nms)

    # Remove borders
    h, w = gray1.shape
    keypoints = np.array([pt for pt in keypoints if kernel_size < pt[0] < h - kernel_size and kernel_size < pt[1] < w - kernel_size])

    # Add grid points
    grid_points = []
    for y in range(grid_stride, h - grid_stride, grid_stride):
        for x in range(grid_stride, w - grid_stride, grid_stride):
            grid_points.append([y, x])

    if keypoints.size > 0:
        keypoints = np.vstack((keypoints, grid_points))
    else:
        keypoints = np.array(grid_points)

    return keypoints

def generate_motion_mask(image_shape, keypoints, arrow_length=10, intensity=255):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for y, x in keypoints:
        end_y = max(0, y - arrow_length)
        cv2.arrowedLine(mask, (x, y), (x, end_y), color=intensity, thickness=1, tipLength=0.4)
    return mask

def rle_encode(mask):
    flat = mask.flatten()
    rle = []
    prev = flat[0]
    count = 1
    for val in flat[1:]:
        if val == prev:
            count += 1
        else:
            rle.append((int(prev), count))
            prev = val
            count = 1
    rle.append((int(prev), count))
    return rle

# --- Main Execution ---
# Load two frames (change paths as needed)
img1 = cv2.imread("data/UVG/images/original/Beauty/im00001.png")
img2 = cv2.imread("data/UVG/images/original/Beauty/im00019.png")

img1 = cv2.resize(img1, (384, 384))
img2 = cv2.resize(img2, (384, 384))

# Compute keypoints
keypoints = compute_sparse_motion_guidance(img1, img2)

# Generate motion mask
motion_mask = generate_motion_mask(img1.shape, keypoints)

# Encode + Compress
rle_data = rle_encode(motion_mask)
compressed_data = zlib.compress(pickle.dumps(rle_data), level=9)

# Save compressed binary file
with open("mask_rle_zlib.bin", "wb") as f:
    f.write(compressed_data)

# Save visualization
plt.figure(figsize=(8, 8), dpi=300)
plt.imshow(motion_mask, cmap="gray")
plt.title("Sparse Motion Mask")
plt.axis("off")
plt.tight_layout()
plt.savefig("sparse_motion_mask.png")
print("Saved compressed mask and plot.")
