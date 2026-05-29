import os

# Suppress TensorFlow INFO/WARNING logs and oneDNN messages before TF import.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import matplotlib.pyplot as plt
from src import load_model, grad_cam, save_overlay, save_heatmap, compute_heatmap_metrics
from src.model import load_and_preprocess, predict_top3

# --- Parameters ---
IMG_PATH   = 'cat.jpg'
LAYER_NAME = 'block5_conv3'

# --- Pipeline ---
model               = load_model()
orig_img, arr       = load_and_preprocess(IMG_PATH)
top3                = predict_top3(model, arr)
heatmap             = grad_cam(model, arr, LAYER_NAME)

os.makedirs('figures', exist_ok=True)
save_overlay(heatmap, orig_img, top3)
save_heatmap(heatmap, orig_img, top3)

metrics = compute_heatmap_metrics(heatmap)

# --- Output ---
print('\nTop-3 predictions:')
for i, (imagenet_id, label, prob) in enumerate(top3, start=1):
    print(f'  {i}. {label} ({imagenet_id}) — {prob:.2%}')

print('\nHeatmap metrics:')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
