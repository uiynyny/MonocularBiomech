import mujoco
import numpy as np
from typing import List, Dict
from jaxtyping import Float, Array
from tqdm import trange, tqdm


# use %env MUJOCO_GL=egl to avoid the need for a display
def render_trajectory(
    pose: Float[Array, "time n_pose"] | List[Dict],
    filename: str = None,
    mj_model: mujoco.MjModel = None,
    xml_path: str = None,
    body_scale: Float[Array, "nscale 1"] | None = None,
    site_offsets: np.array = None,
    margin: float | None = None,
    heel_vertical_offset: float | None = None,
    height: int = 480,
    width: int = 240,
    fps: int = 30,
    show_grfs: bool = False,
    actuators: Float[Array, "time n_actuators"] | None = None,
    azimuth: float = 135,
    blank_background: None | int = None,
    shadow=True,
    hide_tendon=False,
):
    """
    Renders a trajectory of poses using mujoco.

    Args:
        pose: trajectory of poses (or list of mjData)
        filename: if specified, saves the video to this file
        xml_path: path to the xml file (optional)
        site_offsets: offsets for the sites
        height: height of the video
        width: width of the video
        fps: frames per second
        show_grfs: whether to show ground reaction forces
        azimuth: azimuth of the camera
        blank_background: if specified, sets the background to this color (255 is white, 0 is black)
        shadow: whether to show shadows

    Returns:
        if filename is None, returns the images as a list
    """

    from monocular_demos.biomechanics_mjx.forward_kinematics import (
        ForwardKinematics,
        offset_sites,
        scale_model,
        set_margin,
        shift_geom_vertically,
    )
    import numpy as np

    fk = ForwardKinematics(xml_path=xml_path)

    if mj_model is not None:
        print("Using provided mj_model")
        model = mj_model
    else:
        model = fk.model  # non-mjx model

        if margin is not None:
            model = set_margin(model, margin)

        if heel_vertical_offset is not None:
            heel_geom_names = [
                "l_foot_col1",
                "l_foot_col3",
                "r_foot_col1",
                "r_foot_col3",
            ]
            heel_idx = np.array([fk.geom_names.index(g) for g in heel_geom_names])
            model = shift_geom_vertically(model, heel_idx, heel_vertical_offset)

    if body_scale is not None:
        scale = 1 + fk.build_default_scale_mixer() @ body_scale
        model = scale_model(model, scale)

    if site_offsets is not None:
        model = offset_sites(model, site_offsets)

    data = mujoco.MjData(model)

    scene_option = mujoco.MjvOption()
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    if show_grfs:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        if actuators is not None:
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True

    if blank_background is not None:
        model.tex_data = np.zeros_like(model.tex_data) + blank_background

    if not shadow:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
        scene_option.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

    if hide_tendon:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False

    # remove weird overlay
    geom_1_indices = np.where(model.geom_group == 1)
    model.geom_rgba[geom_1_indices, 3] = 0

    camera = mujoco.MjvCamera()
    camera.distance = 3
    camera.azimuth = azimuth
    camera.elevation = -20

    renderer = mujoco.Renderer(model, height=height, width=width)

    images = []
    for i in trange(len(pose)):
        if isinstance(pose, list):
            # when we are passed a list of mj_data just use them
            data = pose[i]

            # relight as otherwise super dark
            mujoco.mj_camlight(model, data)

        else:
            data.qpos = pose[i]
            if show_grfs or (actuators is not None):
                mujoco.mj_forward(model, data)
            else:
                mujoco.mj_kinematics(model, data)

        if i == 0:
            camera.lookat = data.xpos[1]
        else:
            camera.lookat = camera.lookat * 0.7 + data.xpos[1] * 0.3

        if actuators is not None:
            data.ctrl = data.ctrl * 0.0
            data.act = actuators[i]

        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        images.append(renderer.render())

    if filename is not None:
        import cv2
        import numpy as np

        cap = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        for i in trange(len(images)):
            # fix the color channels
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            cap.write(img)

        cap.release()

    else:
        return images


def render_tiled(
    model: mujoco.MjModel,
    total_frames: int,
    qpos: List[Float[Array, "time n_pose"]],
    filename: str = None,
    image_size: tuple = (1440, 960),
    panel_size: tuple = (240, 480),
):
    """
    Renders a tiled video of multiple trajectories.

    Args:
        total_frames: number of frames to render
        qpos: list of trajectories to render
        filename: if specified, saves the video to this file
        image_size: size of the final image
        panel_size: size of each panel

    Returns:
        images: list of images
    """
    from monocular_demos.biomechanics_mjx import ForwardKinematics
    from monocular_demos.biomechanics_mjx.forward_kinematics import offset_sites

    data = mujoco.MjData(model)

    scene_option = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    camera.distance = 3
    camera.azimuth = 155
    camera.elevation = -20

    n_height = image_size[1] // panel_size[1]
    n_width = image_size[0] / panel_size[0]
    vids_per_frame = (np.ceil(n_width).astype(int) + 1) * n_height
    n_videos = (len(qpos) // n_height) * n_height

    video_fps = 30

    horizontal_scrolling_px_sec = panel_size[0] / 6

    geom_1_indices = np.where(model.geom_group == 1)
    model.geom_rgba[geom_1_indices, 3] = 0

    renderer = mujoco.Renderer(model, height=panel_size[1], width=panel_size[0])

    def render_pose_idx(pose, idx):
        idx = idx % len(pose)
        data.qpos = pose[idx]
        mujoco.mj_kinematics(model, data)
        camera.lookat = data.xpos[1]
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        return renderer.render()

    def render_frame(idx):
        horizontal_offset_px = (idx / video_fps) * horizontal_scrolling_px_sec
        column_offset = horizontal_offset_px / panel_size[0]
        first_column = np.floor(column_offset).astype(int)
        frame_offset_left = int(np.mod(column_offset, 1) * panel_size[0])
        frame_offset_right = int(frame_offset_left + image_size[0])

        video_idx = np.arange(
            first_column * n_height, first_column * n_height + vids_per_frame
        )
        video_idx = np.mod(video_idx, n_videos)

        images = [render_pose_idx(qpos[i], idx) for i in video_idx]
        # tile the images as n_height x n_width, going down first (requires the transpose)
        images = np.array(images).reshape(-1, n_height, panel_size[1], panel_size[0], 3)
        images = np.transpose(images, (1, 0, 2, 3, 4))

        rows = [np.concatenate(images[i].tolist(), axis=1) for i in range(n_height)]
        tiled_images = np.concatenate(rows, axis=0)

        return tiled_images[:, frame_offset_left:frame_offset_right]

    images = [render_frame(i) for i in tqdm(range(0, total_frames))]

    if filename is not None:
        # use cv2 to write the video
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            filename, fourcc, video_fps, (image_size[0], image_size[1])
        )
        for image in tqdm(images):
            # convert colorspace
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            video.write(image)
        video.release()

    return images


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    plot_radius = min([plot_radius, 1000])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def render_trajectory_keypoints(
    k3d: Float[Array, "time n_keypoints 3"],
    k3d_gt: Float[Array, "time n_keypoints 3"] = None,
    gt_confidence: Float[Array, "time n_keypoints"] = None,
    confidence_threshold=0.1,
    filename: str = None,
    fps: int = 30,
):
    """
    Create a 3D scatter plot time series video from a numpy array.

    Parameters:
    k3d (np.ndarray): 3D keypoints
    k3d_gt (np.ndarray): An optional second set of 3D keypoints (displayed in black).
    filename (str): The name of the output video file.
    fps (int): Frames per second for the video.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    if k3d_gt is not None:
        assert (
            k3d.shape[0] == k3d_gt.shape[0]
        ), "The number of frames must be the same for both sets of keypoints."
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Set up the writer
    metadata = dict(
        title="3D Scatter Plot Time Series",
        artist="Matplotlib",
        comment="Scatter plot animation",
    )
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    with writer.saving(fig, filename, dpi=100):
        for t in tqdm(range(k3d.shape[0])):
            ax.cla()  # Clear the plot
            ax.scatter(k3d[t, :, 0], k3d[t, :, 1], k3d[t, :, 2], c="blue")
            if k3d_gt is not None:
                # only plot confident points
                if gt_confidence is not None:
                    confident_points = gt_confidence[t] > confidence_threshold
                    ax.scatter(
                        k3d_gt[t, confident_points, 0],
                        k3d_gt[t, confident_points, 1],
                        k3d_gt[t, confident_points, 2],
                        c="black",
                    )
                else:
                    ax.scatter(
                        k3d_gt[t, :, 0], k3d_gt[t, :, 1], k3d_gt[t, :, 2], c="black"
                    )
                # my_max = k3d.max(axis=1).max(axis=0)
                # ax.scatter(my_max[0],my_max[1],my_max[2],c='r')
                # my_min = k3d.min(axis=1).min(axis=0)
                # ax.scatter(my_min[0],my_min[1],my_min[2])

            set_axes_equal(ax)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Time: {t}")
            if t == 0:
                fig.legend(["Predicted", "Ground Truth"], loc="upper right")

            writer.grab_frame()
    plt.close(fig)


def jupyter_embed_video(video_filename: str):

    from IPython.display import HTML
    import subprocess
    import tempfile
    from base64 import b64encode
    import os

    # get temporary mp4 output
    fid, fn = tempfile.mkstemp(suffix=".mp4")
    # close fid
    os.close(fid)

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_filename, "-hide_banner", "-loglevel", "error", fn]
    )

    video = open(fn, "rb").read()
    video_encoded = b64encode(video).decode("ascii")
    video_tag = f'<video controls src="data:video/mp4;base64,{video_encoded}">'

    os.remove(fn)

    return HTML(video_tag)
