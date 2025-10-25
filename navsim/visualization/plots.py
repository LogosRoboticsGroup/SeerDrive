from typing import Any, Callable, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import io
from typing import Dict, Union, List
from matplotlib import cm

from navsim.common.dataclasses import Scene
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG
from navsim.agents.abstract_agent import AbstractAgent
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import (
    add_annotations_to_camera_ax,
    add_lidar_to_camera_ax,
    add_camera_ax,
)
import numpy as np

print('larger figure margin to 128')
BEV_PLOT_CONFIG["figure_margin"] = (128, 128)

def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def configure_all_ax(ax: List[List[plt.Axes]]) -> List[List[plt.Axes]]:
    """
    Iterates through 2D ax list/array to apply configurations
    :param ax: 2D list/array of matplotlib ax object
    :return: configure axes
    """
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            configure_ax(ax[i][j])

    return ax


def plot_bev_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, plt.Axes]:
    """
    General plot for birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_agent(scene: Scene, agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory()
    require_scene = agent.requires_scene if hasattr(agent, "requires_scene") else False
    if require_scene:
        agent_trajectory = agent.compute_trajectory(scene.get_agent_input(), scene)
    else:
        agent_trajectory = agent.compute_trajectory(scene.get_agent_input())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax

TAB_10: Dict[int, str] = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
    8: "#bcbd22",
    9: "#17becf",
}


NEW_TAB_10: Dict[int, str] = {
    0: "#4e79a7",  # blue
    1: "#f28e2b",  # orange
    2: "#e15759",  # red
    3: "#76b7b2",  # cyan
    4: "#59a14f",  # green
    5: "#edc948",  # yellow
    6: "#b07aa1",  # violet
    7: "#ff9da7",
    8: "#9c755f",
    9: "#bab0ac",
}


ELLIS_5: Dict[int, str] = {
    0: "#DE7061",  # red
    1: "#B0E685",  # green
    2: "#4AC4BD",  # cyan
    3: "#E38C47",  # orange
    4: "#699CDB",  # blue
}

TRAJECTORY_WITH_CANDIDATES_CONFIG: Dict[str, Any] = {
    "human": {
        "fill_color": NEW_TAB_10[4],
        "fill_color_alpha": 1.0,
        "line_color": NEW_TAB_10[4],
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 5,
        "marker_edge_color": "black",
        "zorder": 3,
    },
    "agent": {
        "fill_color": ELLIS_5[0],
        "fill_color_alpha": 1.0,
        "line_color": ELLIS_5[0],
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 5,
        "marker_edge_color": "black",
        "zorder": 3,
    },
    "candidates": {
        "line_color": "yellow",               
        "line_color_alpha": 0.5,              
        "line_width": 2.0,
        "line_style": "--",
        "zorder": 2,
        "score_threshold": 0.5,               
        "colormap": "plasma",                 
        "line_alpha": 1.0,                    
        "low_score_color": "gray",            
        "low_score_alpha": 0.3,               
    }
}

def plot_bev_with_agent_and_traj_candidates(scene: Scene, agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory()
    agent_trajectory, all_traj_candidates, final_scores, im_rewards = agent.compute_trajectory_with_vis(scene.get_agent_input())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    
    # Add all trajectory candidates to the plot
    add_trajectory_candidates_to_bev_ax(ax, all_traj_candidates, final_scores, TRAJECTORY_WITH_CANDIDATES_CONFIG["candidates"])
    
    # Add human trajectory to the plot
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_WITH_CANDIDATES_CONFIG["human"])
    
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def add_trajectory_candidates_to_bev_ax(
        ax: plt.Axes, 
        all_trajectories: np.ndarray, 
        scores: np.ndarray, 
        config: Dict[str, Any]
    ) -> plt.Axes:
    """
    Add all trajectory candidates to the plot, with color intensity representing their scores.
    :param ax: matplotlib ax object
    :param all_trajectories: numpy array of shape [256, 8, 3] containing all trajectory candidates
    :param scores: numpy array of shape [256] containing scores for each trajectory
    :param config: dictionary with plot parameters for 'candidates'
    :return: ax with plot
    """

    threshold = config.get("score_threshold", 0.0)
    
    valid_indices = scores > threshold
    if not np.any(valid_indices):
        print("No trajectories exceed the score threshold.")
        return ax

    valid_scores = scores[valid_indices]
    normalized_scores = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
    
    cmap_name = config.get("colormap", "viridis")
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=valid_scores.min(), vmax=valid_scores.max())
    
    for i, trajectory in enumerate(all_trajectories):
        poses = np.concatenate([np.array([[0, 0]]), trajectory[:, :2]])

        if scores[i] > threshold:
            color = cmap(norm(scores[i]))
            normalized_score = (scores[i] - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
            alpha = np.clip(normalized_score * config.get("line_alpha", 1.0), 0, 1)
        else:
            color = "#d3d3d3"  # 淡灰色
            alpha = 0.4  # 更低的透明度
        
        ax.plot(
            poses[:, 1],
            poses[:, 0],
            color=color,
            alpha=alpha,
            linewidth=config.get("line_width", 1.0),
            linestyle=config.get("line_style", "-"),
            zorder=config.get("zorder", 1),
        )
    
    if np.any(valid_indices):
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Trajectory Score')
    
    return ax


def plot_cameras_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_camera_ax(ax[0, 0], frame.cameras.cam_l0)
    add_camera_ax(ax[0, 1], frame.cameras.cam_f0)
    add_camera_ax(ax[0, 2], frame.cameras.cam_r0)

    add_camera_ax(ax[1, 0], frame.cameras.cam_l1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_camera_ax(ax[1, 2], frame.cameras.cam_r1)

    add_camera_ax(ax[2, 0], frame.cameras.cam_l2)
    add_camera_ax(ax[2, 1], frame.cameras.cam_b0)
    add_camera_ax(ax[2, 2], frame.cameras.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_lidar(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the lidar pc) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_lidar_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.lidar)

    add_lidar_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.lidar)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_lidar_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.lidar)

    add_lidar_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.lidar)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the bounding boxes) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_annotations_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.annotations)

    add_annotations_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.annotations)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_annotations_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.annotations)

    add_annotations_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.annotations)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def frame_plot_to_pil(
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []

    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        fig, ax = callable_frame_plot(scene, frame_idx)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images


def frame_plot_to_gif(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
    duration: float = 500,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param file_name: file path for saving to save
    :param duration: frame interval in ms, defaults to 500
    """
    images = frame_plot_to_pil(callable_frame_plot, scene, frame_indices)
    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)
