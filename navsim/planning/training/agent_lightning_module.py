import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Tuple
import torch
from navsim.agents.abstract_agent import AbstractAgent

class AgentLightningModule(pl.LightningModule):
    def __init__(
        self,
        agent: AbstractAgent,
    ):
        super().__init__()
        self.agent = agent

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ):
        features, targets = batch

        input_target = self.agent.config.input_target if hasattr(self.agent.config, 'input_target') else False
        if input_target:
            prediction = self.agent.forward(features, targets)
        else:
            prediction = self.agent.forward(features)

        loss_dict = self.agent.compute_loss(features, targets, prediction)
        if isinstance(loss_dict, Tensor):
            loss_dict = {"traj_loss": loss_dict}
            
        total_loss = 0.0
        for loss_key, loss_value in loss_dict.items():
            self.log(f"{logging_prefix}_{loss_key}", loss_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            if 'acc' in loss_key:
                continue
            total_loss = total_loss + loss_value
        self.log(f"{logging_prefix}_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss
    
    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()
    
    # def backward(self, loss):
    #     # print('set detect anomaly')
    #     # torch.autograd.set_detect_anomaly(True)
    #     loss.backward(retain_graph=True)
