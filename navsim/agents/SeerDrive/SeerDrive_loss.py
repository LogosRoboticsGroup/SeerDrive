from typing import Dict
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Dict, Any


def compute_SeerDrive_loss(
    targets,
    predictions,
    config,
) -> torch.Tensor:
    loss_dict = {}
    # offset regression
    traj_offset_loss = compute_traj_offset_loss(predictions, targets, config)
    loss_dict['traj_offset_loss'] = traj_offset_loss * config.traj_offset_loss_weight

    # offset reward
    trajectory_anchors = predictions["trajectory_anchors"]  # torch.Size([256, 8, 3])
    imitation_rewards = predictions["trajectory_offset_rewards"]  # torch.Size([B, 256])
    offset_im_reward_loss = compute_im_reward_loss(targets, imitation_rewards, trajectory_anchors)
    loss_dict['offset_im_reward_loss'] = offset_im_reward_loss * config.offset_im_reward_weight

    ######
    predictions["trajectory_offset"] = predictions["trajectory_offset_refine"]
    traj_offset_loss = compute_traj_offset_loss(predictions, targets, config)
    loss_dict['traj_offset_loss_refine'] = traj_offset_loss * config.traj_offset_loss_weight * 0.5

    trajectory_anchors = predictions["trajectory_anchors"]  # torch.Size([256, 8, 3])
    imitation_rewards = predictions["trajectory_offset_rewards_refine"]  # torch.Size([B, 256])
    offset_im_reward_loss = compute_im_reward_loss(targets, imitation_rewards, trajectory_anchors)
    loss_dict['offset_im_reward_loss_refine'] = offset_im_reward_loss * config.offset_im_reward_weight * 0.5

    predictions["trajectory_offset"] = predictions["trajectory_offset_refine_refine"]
    traj_offset_loss = compute_traj_offset_loss(predictions, targets, config)
    loss_dict['traj_offset_loss_refine_refine'] = traj_offset_loss * config.traj_offset_loss_weight

    trajectory_anchors = predictions["trajectory_anchors"]  # torch.Size([256, 8, 3])
    imitation_rewards = predictions["trajectory_offset_rewards_refine_refine"]  # torch.Size([B, 256])
    offset_im_reward_loss = compute_im_reward_loss(targets, imitation_rewards, trajectory_anchors)
    loss_dict['offset_im_reward_loss_refine_refine'] = offset_im_reward_loss * config.offset_im_reward_weight
    ######

    # im & sim rewards
    trajectory_anchors = predictions["trajectory_anchors"]  # torch.Size([256, 8, 3])
    imitation_rewards = predictions["im_rewards"]  # torch.Size([B, 256])
    im_reward_loss = compute_im_reward_loss(targets, imitation_rewards, trajectory_anchors)

    sim_rewards = predictions["sim_rewards"]  # torch.Size([B, 5, 256])
    sim_reward_loss = compute_sim_reward_loss(targets, sim_rewards)

    loss_dict['im_reward_loss'] = im_reward_loss
    loss_dict['sim_reward_loss'] = sim_reward_loss

    use_agent_loss = config.use_agent_loss if hasattr(config, "use_agent_loss") else True
    if use_agent_loss:
        agent_cls_loss, agent_box_loss = compute_SeerDrive_agent_loss(
                                            targets["agent_states"], 
                                            targets["agent_labels"],
                                            predictions["agent_states"],
                                            predictions["agent_labels"],
                                            config,
                                            )
        agent_class_weight = config.agent_class_weight if hasattr(config, "agent_class_weight") else 0.0
        agent_box_weight = config.agent_box_weight if hasattr(config, "agent_box_weight") else 0.0

        loss_dict['agent_cls_loss'] = agent_cls_loss * agent_class_weight
        loss_dict['agent_box_loss'] = agent_box_loss * agent_box_weight

    if config.use_map_loss:
        use_focal_loss_for_map = config.use_focal_loss_for_map if hasattr(config, 'use_focal_loss_for_map') else False
        focal_loss_alpha = config.focal_loss_alpha if hasattr(config, 'focal_loss_alpha') else 0.25
        focal_loss_gamma = config.focal_loss_gamma if hasattr(config, 'focal_loss_gamma') else 2.0
        focal_loss_fn = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        map_loss = focal_loss_fn(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
        loss_dict['map_loss'] = map_loss * config.bev_semantic_weight

        # fut map
        bz, num_trajs, h, w = targets["fut_bev_semantic_map"].shape
        gt_fut_bev_semantic_map = targets["fut_bev_semantic_map"].reshape(bz * num_trajs, h, w)
        focal_loss_fn = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        fut_map_loss = focal_loss_fn(predictions["fut_bev_semantic_map"], gt_fut_bev_semantic_map.long())
        loss_dict['fut_map_loss'] = fut_map_loss * config.fut_bev_semantic_weight

    return loss_dict


def compute_traj_offset_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: Any
) -> torch.Tensor:
    """
    Computes the trajectory offset loss using a Winner-Take-All (WTA) strategy.

    Args:
        predictions (Dict[str, torch.Tensor]): Model predictions containing:
            - "trajectory_anchors": Tensor of shape (batch_size, 256, traj_dim)
            - "trajectory_offset": Tensor of shape (batch_size, 256, traj_dim)
        targets (Dict[str, torch.Tensor]): Ground truth data containing:
            - "trajectory": Tensor of shape (batch_size, traj_dim)
        config (Any): Configuration object with the following attributes:
            - use_traj_offset (bool): Whether to compute the trajectory offset loss.
            - traj_offset_loss_weight (float): Weight for the trajectory offset loss.

    Returns:
        torch.Tensor: Weighted trajectory offset loss. Returns zero tensor if `use_traj_offset` is False.
    """
    # Extract necessary tensors from predictions and targets
    trajectory_anchors = predictions["trajectory_anchors"]       
    # Shape: (batch_size, 256, traj_dim) torch.Size([256, 8, 3])
    predicted_offsets = predictions["trajectory_offset"]      
    # Shape: (batch_size, 256, traj_dim) torch.Size([B, 256, 8, 3])
    gt_trajectories = targets["trajectory"].squeeze(1)     
    # Shape: (batch_size, traj_dim) torch.Size([B, 8, 3])

    bz = gt_trajectories.shape[0]
    num_trajs = trajectory_anchors.shape[0]
    gt_trajectories = gt_trajectories.reshape(bz, -1)
    predicted_offsets = predicted_offsets.reshape(bz, num_trajs, -1)
    trajectory_anchors = trajectory_anchors.unsqueeze(0).repeat(bz, 1, 1, 1).reshape(bz, num_trajs, -1)

    # Ensure that the batch sizes match
    batch_size, num_clusters, traj_dim = trajectory_anchors.shape
    assert gt_trajectories.shape == (batch_size, traj_dim), \
    f"Ground truth trajectories shape {gt_trajectories.shape} does not match expected {(batch_size, traj_dim)}"

    # Expand ground truth trajectories for broadcasting: (batch_size, 1, traj_dim)
    gt_expanded = gt_trajectories.unsqueeze(1)            # Shape: (batch_size, 1, traj_dim)

    # Compute L2 distances between each cluster center and the ground truth trajectory
    # Resulting shape: (batch_size, 256)
    distances = torch.norm(trajectory_anchors - gt_expanded, dim=2)  # Euclidean distance

    # Find the index of the closest cluster center for each sample in the batch
    # Shape: (batch_size,)
    min_indices = torch.argmin(distances, dim=1)            # Winner cluster index per sample

    # Create a batch index tensor for advanced indexing
    batch_indices = torch.arange(batch_size, device=trajectory_anchors.device)  # Shape: (batch_size,)

    # Gather the selected (winner) cluster centers based on min_indices
    # Shape: (batch_size, traj_dim)
    selected_trajectory_anchors = trajectory_anchors[batch_indices, min_indices]

    # Compute the ground truth offsets: gt_trajectory - selected_trajectory_anchor
    # Shape: (batch_size, traj_dim)
    gt_offsets = gt_trajectories - selected_trajectory_anchors

    # Gather the predicted offsets corresponding to the selected (winner) cluster centers
    # Shape: (batch_size, traj_dim)
    selected_predicted_offsets = predicted_offsets[batch_indices, min_indices]

    # Compute the L1 loss (Mean Absolute Error) between predicted offsets and ground truth offsets
    # Reduction is 'mean' to average over all samples and trajectory dimensions
    traj_offset_loss = F.l1_loss(selected_predicted_offsets, gt_offsets, reduction='mean')

    return traj_offset_loss


# 这个函数用于end point预测计算loss
def compute_traj_offset_loss_endpoint(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: Any
) -> torch.Tensor:
    """
    Computes the trajectory offset loss using a Winner-Take-All (WTA) strategy.

    Args:
        predictions (Dict[str, torch.Tensor]): Model predictions containing:
            - "trajectory_anchors": Tensor of shape (batch_size, 256, traj_dim)
            - "trajectory_offset": Tensor of shape (batch_size, 256, traj_dim)
        targets (Dict[str, torch.Tensor]): Ground truth data containing:
            - "trajectory": Tensor of shape (batch_size, traj_dim)
        config (Any): Configuration object with the following attributes:
            - use_traj_offset (bool): Whether to compute the trajectory offset loss.
            - traj_offset_loss_weight (float): Weight for the trajectory offset loss.

    Returns:
        torch.Tensor: Weighted trajectory offset loss. Returns zero tensor if `use_traj_offset` is False.
    """
    # Extract necessary tensors from predictions and targets
    trajectory_anchors = predictions["trajectory_anchors"][:, -1].unsqueeze(1)       
    # Shape: (batch_size, 256, traj_dim) torch.Size([256, 1, 3])
    predicted_offsets = predictions["trajectory_offset"]      
    # Shape: (batch_size, 256, traj_dim) torch.Size([B, 256, 1, 3])
    gt_trajectories = targets["trajectory"].squeeze(1)[:, -1].unsqueeze(1)     
    # Shape: (batch_size, traj_dim) torch.Size([B, 1, 3])

    bz = gt_trajectories.shape[0]
    num_trajs = trajectory_anchors.shape[0]
    gt_trajectories = gt_trajectories.reshape(bz, -1)
    predicted_offsets = predicted_offsets.reshape(bz, num_trajs, -1)
    trajectory_anchors = trajectory_anchors.unsqueeze(0).repeat(bz, 1, 1, 1).reshape(bz, num_trajs, -1)

    # Ensure that the batch sizes match
    batch_size, num_clusters, traj_dim = trajectory_anchors.shape
    assert gt_trajectories.shape == (batch_size, traj_dim), \
    f"Ground truth trajectories shape {gt_trajectories.shape} does not match expected {(batch_size, traj_dim)}"

    # Expand ground truth trajectories for broadcasting: (batch_size, 1, traj_dim)
    gt_expanded = gt_trajectories.unsqueeze(1)            # Shape: (batch_size, 1, traj_dim)

    # Compute L2 distances between each cluster center and the ground truth trajectory
    # Resulting shape: (batch_size, 256)
    distances = torch.norm(trajectory_anchors - gt_expanded, dim=2)  # Euclidean distance

    # Find the index of the closest cluster center for each sample in the batch
    # Shape: (batch_size,)
    min_indices = torch.argmin(distances, dim=1)            # Winner cluster index per sample

    # Create a batch index tensor for advanced indexing
    batch_indices = torch.arange(batch_size, device=trajectory_anchors.device)  # Shape: (batch_size,)

    # Gather the selected (winner) cluster centers based on min_indices
    # Shape: (batch_size, traj_dim)
    selected_trajectory_anchors = trajectory_anchors[batch_indices, min_indices]

    # Compute the ground truth offsets: gt_trajectory - selected_trajectory_anchor
    # Shape: (batch_size, traj_dim)
    gt_offsets = gt_trajectories - selected_trajectory_anchors

    # Gather the predicted offsets corresponding to the selected (winner) cluster centers
    # Shape: (batch_size, traj_dim)
    selected_predicted_offsets = predicted_offsets[batch_indices, min_indices]

    # Compute the L1 loss (Mean Absolute Error) between predicted offsets and ground truth offsets
    # Reduction is 'mean' to average over all samples and trajectory dimensions
    traj_offset_loss = F.l1_loss(selected_predicted_offsets, gt_offsets, reduction='mean')

    return traj_offset_loss


def compute_im_reward_loss(
    targets: Dict[str, torch.Tensor],
    prediction_rewards,  # torch.Size([B, 256])
    trajectory_anchors,  # torch.Size([256, 8, 3]) 
) -> torch.Tensor:
    Bz = targets["trajectory"].shape[0]
    # Get target trajectory
    target_trajectory = targets["trajectory"][:, -1]  # the last frame
    target_trajectory = target_trajectory.reshape(Bz, -1).unsqueeze(1).float()

    # Calculate L2 distance between each of the 256 predefined trajectories and the target trajectory
    num_trajs = trajectory_anchors.shape[0]
    trajectory_anchors = trajectory_anchors.reshape(num_trajs, -1).unsqueeze(0).repeat(Bz, 1, 1).to(target_trajectory.device)
    l2_distances = torch.cdist(trajectory_anchors, target_trajectory, p=2)  # Shape: [batch_size, 256]
    l2_distances = l2_distances.squeeze(-1)

    # Apply softmax to L2 distances to get reward targets
    reward_targets = torch.softmax(-l2_distances, dim=-1)  # Shape: [batch_size, 256]

    # Compute loss using cross-entropy
    prediction_rewards = prediction_rewards.squeeze(-1).clamp(1e-6, 1 - 1e-6)
    im_reward_loss = -torch.sum(reward_targets * prediction_rewards.log()) / Bz

    return im_reward_loss


def compute_sim_reward_loss(
    targets: Dict[str, torch.Tensor],
    predicted_rewards: torch.Tensor,
) -> torch.Tensor:
    epsilon = 1e-6
    # Load precomputed target rewards
    batch_size = targets['sim_reward'].shape[0]
    target_rewards = targets['sim_reward'][:, -1] # the last frame torch.Size([B, 5, 256])

    # Compute loss using binary cross-entropy # 5 is the number of metrics
    sim_reward_loss = -torch.mean(
        target_rewards * (predicted_rewards + epsilon).log() + (1 - target_rewards) * (1 - predicted_rewards + epsilon).log()
    ) * 5

    return sim_reward_loss


def compute_SeerDrive_agent_loss(
    gt_states, 
    gt_valid,
    pred_states,
    pred_logits,
    config,
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    # gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    # pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    # visualize_agent_predictions(pred_states, gt_states, pred_logits, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    agent_class_weight = config.agent_class_weight if hasattr(config, "agent_class_weight") else 0.0
    agent_box_weight = config.agent_box_weight if hasattr(config, "agent_box_weight") else 0.0
    cost = agent_class_weight * ce_cost + agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', epsilon=1e-6):
        """
        :param alpha: 类别权重, 可以是 float 或包含每个类别权重的 list (用于类别不均衡).
        :param gamma: 调节因子，控制焦点机制的强度.
        :param reduction: 'none', 'mean', 'sum', 控制输出损失的形式.
        :param epsilon: 防止 pt 过小导致数值不稳定的小常数.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes, ...]
        # targets: [batch_size, ...]

        # 计算普通交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 为了避免 pt 过小而导致数值不稳定，添加 epsilon
        pt = torch.exp(-ce_loss).clamp(min=self.epsilon)  # pt 是 log 的逆，等价于 softmax 输出中正确类别的概率

        # Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 如果指定了 alpha
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):  # 如果 alpha 是单一浮点数
                alpha_t = torch.full_like(targets, fill_value=self.alpha, dtype=torch.float32)
            else:  # 如果 alpha 是类别权重列表
                alpha_t = torch.tensor(self.alpha, dtype=torch.float32).to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss

        # 返回值根据 reduction 参数求和或者求平均
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
