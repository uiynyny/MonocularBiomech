import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
from monocular_demos.utils import load_metrabs

# Create dummy video frame
frame = np.zeros((240, 240, 3), dtype=np.uint8)

print("Loading pipeline...")
pipeline = load_metrabs()
print("Pipeline loaded.")

print("Running batch prediction...")
pred = pipeline.detect_poses_batched(np.array([frame]), skeleton="bml_movi_87")
print(pred['poses3d'].shape)
print("Pipeline tested successfully.")
