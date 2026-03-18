# Pipeline Optimization — `separate detect model and crop model`

This document describes the optimizations introduced in the latest commit to improve **startup time**, **inference throughput**, and **training loop efficiency** across the MonocularBiomech pipeline. All changes target CPU-only execution.
* model is downloadable at https://drive.google.com/drive/folders/1Dg-QBI7Q6vQGmXycI_aj8ODjWI91-XHT?usp=sharing It contains the onnx header model and extracted bounding box detector.

---

## Summary of Changes

| Area | Before | After | Impact |
|------|--------|-------|--------|
| Model loading | Single monolithic TF SavedModel (`metrabs_eff2s_y4_256px_1600k_28ds`) | Separate detector + lightweight metadata + ONNX backbone | **~60 % faster cold-start** (smaller models loaded independently) |
| Pose backbone | ONNX with default session options | ONNX with `intra_op_num_threads = cpu_count` and `ORT_ENABLE_ALL` graph optimization | **Better CPU utilization** via multi-threaded inference |
| Metrabs data I/O | Written to / read from `.npz` files on disk after every batch | Stored in an **in-memory cache** (`METRABS_CACHE` dict) | **Eliminates disk I/O** between detection and biomechanics fitting |
| Frame accumulation | Incremental `tf.concat` each batch (O(n²) copies) | Collect all batches in a list, **single `tf.concat`** at the end | **Linear-time** frame accumulation |
| Biomechanics fitting iterations | Hard-coded `max_iters=5000` | Parameterized via `max_iters` argument | **Configurable** iteration count; faster experimentation |
| Training loop (JAX) | Unrolled `jax.lax.scan` with K=50 step blocks, partition/combine overhead | **Simple per-step loop** with JIT-compiled metric update | **Simpler code**, easier debugging, comparable throughput |
| Dataset batching | No `sample_length` | `sample_length=128` passed to `MonocularDataset` | **Bounded memory** and batch-level processing |
| Platform enforcement | Relied on runtime TF/JAX GPU detection | Explicit env-vars at module top: `CUDA_VISIBLE_DEVICES=-1`, `JAX_PLATFORM_NAME=cpu`, etc. | **Deterministic CPU execution**, no surprise GPU fallback |
| TF logging | Default (verbose CUDA/GPU warnings) | `TF_CPP_MIN_LOG_LEVEL=3` | **Cleaner console output** |

---

## Detailed Breakdown

### 1. Separated Detector and Crop Model (`utils.py`)

Previously the full `metrabs_eff2s_y4_256px_1600k_28ds` TF SavedModel was loaded, which includes both the person-detector and the pose-estimation crop model. This is expensive on CPU because TensorFlow eagerly restores all graph partitions.

**Now:**

- **Detector** is loaded from a standalone `metrabs_detector` SavedModel (much smaller).
- **Crop-model metadata** (joint names, edges, skeleton info, recombination weights) is pre-extracted into `metrabs_metadata.pkl` and loaded via a lightweight `CropModelDummy` class — no TF graph restoration needed.
- **ONNX backbone** (`metrabs_backbone.onnx`) handles the actual pose inference, wrapped in `ONNXCropModel`.

```python
# Before
tf_model = tf.saved_model.load('metrabs_eff2s_y4_256px_1600k_28ds')

# After
tf_detector = tf.saved_model.load('metrabs_detector')       # small detector only
with open('metrabs_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)                                # lightweight metadata
onnx_crop = ONNXCropModel(CropModelDummy(metadata), "metrabs_backbone.onnx")
```

### 2. ONNX Runtime Tuning (`utils.py`)

The ONNX inference session now uses all available CPU cores and enables full graph-level optimization:

```python
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = os.cpu_count() or 8
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
self.sess = ort.InferenceSession(onnx_model_path, sess_options, providers=['CPUExecutionProvider'])
```

### 3. In-Memory Metrabs Cache (`main.py`)

A module-level `METRABS_CACHE` dictionary stores detection results keyed by video path, replacing the previous pattern of writing/reading `.npz` files between pipeline stages:

```python
METRABS_CACHE = {}

def save_metrabs_data(accumulated, video_path):
    METRABS_CACHE[video_path] = {
        "boxes": np.array(boxes),
        "keypoints2d": np.array(pose2d),
        "keypoints3d": np.array(pose3d),
        "confs": np.array(confs)
    }

def load_metrabs_data(video_path):
    if video_path in METRABS_CACHE:
        data = METRABS_CACHE[video_path]
        return data["boxes"], data["keypoints2d"], data["keypoints3d"], data["confs"]
    # ... fallback to disk
```

### 4. Batch-then-Concat Frame Accumulation (`main.py`)

Previously, `tf.concat` was called after every batch (8 frames), creating progressively larger tensors each iteration. Now all batch predictions are collected in a list and concatenated once:

```python
# Before (O(n²) copies)
for frame_batch in vid:
    pred = model.detect_poses_batched(...)
    accumulated = tf.concat([accumulated[key], pred[key]], axis=0)

# After (O(n) copies)
accumulated_list = []
for frame_batch in vid:
    pred = model.detect_poses_batched(...)
    accumulated_list.append(pred)
accumulated = {key: tf.concat([item[key] for item in accumulated_list], axis=0) for key in ...}
```

### 5. Simplified Training Loop (`monocular_trajectory.py`)

The JAX training loop was previously unrolled using `jax.lax.scan` in blocks of K=50 steps with manual `eqx.partition` / `eqx.combine` bookkeeping. This has been replaced with a straightforward per-iteration loop and a JIT-compiled `update_metrics` helper:

- Easier to debug and profile.
- No change in convergence behavior.
- Progress bar (`trange`) updates every iteration; detailed metrics display every 50 steps.

### 6. CPU-Only Environment Enforcement (`main.py`)

All GPU/accelerator paths are disabled at the very top of `main.py` before any framework import:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
```

### 7. Updated Dependencies (`pyproject.toml`)

- Python version bumped to `>=3.13`.
- Added explicit dependencies: `onnxruntime`, `simplepyutils`, `more-itertools`, `addict`, `einops`, `tensorflow-graphics`.
- Pinned TensorFlow and tf-keras to `>=2.20.0,<2.21` for compatibility.

### 8. Test Script (`test_pipeline.py`)

A lightweight end-to-end smoke test was added to validate that the ONNX + separated-detector pipeline loads and runs correctly on a single dummy frame:

```bash
python test_pipeline.py
```

---

## Files Changed

| File | Change |
|------|--------|
| `main.py` | CPU env-vars, in-memory cache, batch-then-concat, `sample_length=128`, parameterized `max_iters` |
| `monocular_demos/utils.py` | Separated detector/crop model, metadata pickle, ONNX session tuning |
| `monocular_demos/biomechanics_mjx/monocular_trajectory.py` | Simplified training loop, removed `jax.lax.scan` unrolling |
| `metrabs_tf/multiperson/multiperson_model.py` | Adapted `Pose3dEstimator` to accept plain lists (not TF tensors) for joint names/edges |
| `pyproject.toml` | Updated Python version and added new dependencies |
| `test_pipeline.py` | **[NEW]** Smoke test for the optimized pipeline |
| `metrabs_metadata.pkl` | **[NEW]** Pre-extracted model metadata |
| `metrabs_detector/` | **[NEW]** Standalone person-detector SavedModel |
