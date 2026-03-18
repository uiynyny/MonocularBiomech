import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import onnxruntime as ort
import simplepyutils as spu
import pickle
from metrabs_tf import tfu3d
from metrabs_tf.multiperson.multiperson_model import Pose3dEstimator

def jax_memory_limit():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def tensorflow_memory_limit():
    # limit tensorflow memory. there are also other approaches
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def video_reader(filename: str, batch_size: int = 8, width: int | None = None):
    """
    Read a video file and yield frames in batches.

    In theory, tensorflow_io has tools for this but they don't seem to work for me. That
    is probably more efficient if it works as they can prefetch. This also will optionally
    downsample the video if compute is a limit.

    Args:
        filename: (str) The path to the video file.
        batch_size: (int) The number of frames to yield at once.
        width: (int | None) The width to downsample to. If None, the original width is used.

    Returns:
        A tuple of (generator, n_frames) where generator yields batches and n_frames is total frame count
    """

    cap = cv2.VideoCapture(filename)
    
    # Get total frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def frame_generator():
        cap = cv2.VideoCapture(filename)  # Create new capture object for generator
        frames = []
        while True:
            ret, frame = cap.read()

            if ret is False:
                if len(frames) > 0:
                    frames = np.array(frames)
                    yield frames
                cap.release()
                return
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if width is not None:
                    # downsample to keep the aspect ratio and output the specified width
                    scale = width / frame.shape[1]
                    height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (width, height))

                frames.append(frame)

                if len(frames) >= batch_size:
                    frames = np.array(frames)
                    yield frames
                    frames = []
    
    cap.release()  # Release the initial capture object
    return frame_generator(), n_frames

def load_metrabs():
    if load_metrabs.model is not None:
        return load_metrabs.model
    

    print("Loading Metadata and Detector...")
    with open('metrabs_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        
    class CropModelDummy:
        def __init__(self, metadata):
            self.input_resolution = metadata['input_resolution']
            self.joint_names = np.array([b.encode('utf8') for b in metadata['joint_names']])
            self.joint_edges = np.array(metadata['joint_edges'])
            if 'recombination_weights' in metadata:
                self.recombination_weights = tf.constant(metadata['recombination_weights'])

    dummy_crop = CropModelDummy(metadata)
    
    tf_detector = tf.saved_model.load('metrabs_detector')
    
    class ONNXCropModel(tf.Module):
        def __init__(self, crop_model, onnx_model_path):
            self.crop_model = crop_model
            self.input_resolution = crop_model.input_resolution
            self.joint_names = crop_model.joint_names
            self.joint_edges = crop_model.joint_edges
            
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = os.cpu_count() or 8
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(onnx_model_path, sess_options, providers=['CPUExecutionProvider'])
            
            # Extract weights for post-processing if any
            if hasattr(crop_model, "recombination_weights"):
                self.recombination_weights = crop_model.recombination_weights

        def backbone_and_head(self, image, training=False):
            # Run the ONNX model which outputs the heatmaps
            def _ort_run(img_np):
                out1, out2 = self.sess.run(None, {'image': img_np.numpy()})
                return out1, out2
            
            head_a, head_b = tf.py_function(_ort_run, [image], [tf.float32, tf.float32])
            
            return None, head_a, head_b

        def latent_points_to_joints(self, points):
            return tfu3d.linear_combine_points(points, self.recombination_weights)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float16),
            tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32)])
        def predict_multi(self, image, intrinsic_matrix):
            # Emulate the call() method of Metrabs crop model
            _, head_a, head_b = self.backbone_and_head(image, False)
            
            # Determine which is coords2d and coords3d based on shape dynamically
            shape_a = tf.shape(head_a)[-1]
            coords2d = tf.cond(tf.equal(shape_a, 2), lambda: head_a, lambda: head_b)
            coords3d = tf.cond(tf.equal(shape_a, 2), lambda: head_b, lambda: head_a)

            # Restore static shape lost during py_function so einops/tf.reshape works downstream
            num_joints = self.joint_names.shape[0]
            coords2d.set_shape([None, num_joints, 2])
            coords3d.set_shape([None, num_joints, 3])
            
            coords3d_abs = tfu3d.reconstruct_absolute(
                coords2d, coords3d, intrinsic_matrix, mix_3d_inside_fov=0.0)

            # If the original model used affine weights, apply them
            if hasattr(self, 'recombination_weights'):
                coords3d_abs = self.latent_points_to_joints(coords3d_abs)
            return coords3d_abs

        @tf.function
        def __call__(self, inp, training=False):
            return self.predict_multi(inp[0], inp[1])
            
    print("Building ONNX Crop Model Wrapper...")
    onnx_crop = ONNXCropModel(dummy_crop, "metrabs_backbone.onnx")
    
    # We need skeleton_infos. They are in the SavedModel. 
    # Pose3dEstimator expects a dict of {name: {'indices': ..., 'names': ..., 'edges': ...}}
    skeleton_infos = {}
    for skel, v in metadata['skeleton_infos'].items():
        skeleton_infos[skel] = {
            'indices': v['indices'],
            'names': v['names'],
            'edges': v['edges']
        }

    # Transform matrix
    joint_transform_matrix = None
    
    # Initialize necessary FLAGS for inference
    for attr, val in [
        ('detector_flip_vertical_too', False),
        ('rot_aug', 0.0),
        ('rot_aug_360', False),
        ('rot_aug_360_half', False),
        ('weak_perspective', False),
        ('stride_train', 32),
        ('centered_stride', False),
        ('proc_side', 256),
        ('mean_relative', True),
    ]:
        if not hasattr(spu.FLAGS, attr):
            setattr(spu.FLAGS, attr, val)

    print("Wiring Pose3dEstimator with ONNX and TensorFlow...")
    pipeline = Pose3dEstimator(
        crop_model=onnx_crop,
        detector=tf_detector,
        skeleton_infos=skeleton_infos,
        joint_transform_matrix=joint_transform_matrix
    )
    
    load_metrabs.model = pipeline
    return pipeline

load_metrabs.model = None

joint_names = [
    "backneck",
    "upperback",
    "clavicle",
    "sternum",
    "umbilicus",
    "lfronthead",
    "lbackhead",
    "lback",
    "lshom",
    "lupperarm",
    "lelbm",
    "lforearm",
    "lwrithumbside",
    "lwripinkieside",
    "lfin",
    "lasis",
    "lpsis",
    "lfrontthigh",
    "lthigh",
    "lknem",
    "lankm",
    "LHeel",
    "lfifthmetatarsal",
    "LBigToe",
    "lcheek",
    "lbreast",
    "lelbinner",
    "lwaist",
    "lthumb",
    "lfrontinnerthigh",
    "linnerknee",
    "lshin",
    "lfirstmetatarsal",
    "lfourthtoe",
    "lscapula",
    "lbum",
    "rfronthead",
    "rbackhead",
    "rback",
    "rshom",
    "rupperarm",
    "relbm",
    "rforearm",
    "rwrithumbside",
    "rwripinkieside",
    "rfin",
    "rasis",
    "rpsis",
    "rfrontthigh",
    "rthigh",
    "rknem",
    "rankm",
    "RHeel",
    "rfifthmetatarsal",
    "RBigToe",
    "rcheek",
    "rbreast",
    "relbinner",
    "rwaist",
    "rthumb",
    "rfrontinnerthigh",
    "rinnerknee",
    "rshin",
    "rfirstmetatarsal",
    "rfourthtoe",
    "rscapula",
    "rbum",
    "Head",
    "mhip",
    "CHip",
    "Neck",
    "LAnkle",
    "LElbow",
    "LHip",
    "LHand",
    "LKnee",
    "LShoulder",
    "LWrist",
    "LFoot",
    "RAnkle",
    "RElbow",
    "RHip",
    "RHand",
    "RKnee",
    "RShoulder",
    "RWrist",
    "RFoot",
]
