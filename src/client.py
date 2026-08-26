from pathlib import Path
import gc
import pickle
import logging

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
# from .losses import FocalLoss
from .losses import *
from .adaptive_gamma import get_client_adaptive_gamma

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None

        # Adaptive gamma information
        self.class_counts = None
        self.entropy = None
        self.gamma = None
        self.adaptive_gamma_config = {}

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client."""

        self.dataloader = DataLoader(
            self.data,
            batch_size=client_config["batch_size"],
            shuffle=True,
            drop_last=True
        )

        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]
        self.criterion_cl = client_config["criterion_cl"]
        self.loss_config = client_config["loss_config"]
        self.focal_loss = FocalLoss()

        # ---------------------------------------------------------
        # Local class-frequency logit adjustment
        # ---------------------------------------------------------
        logit_adjustment_config = client_config.get(
            "logit_adjustment_config",
            {}
        )

        self.logit_adjust_enabled = logit_adjustment_config.get(
            "enabled",
            False
        )

        self.logit_adjust_tau = logit_adjustment_config.get(
            "tau",
            0.5
        )

        if not self.logit_adjust_enabled:
            self.logit_adjust_tau = 0.0

        # ---------------------------------------------------------
        # Adaptive Gamma
        # ---------------------------------------------------------

        self.adaptive_gamma_config = client_config.get(
            "adaptive_gamma_config",
            {}
        )

        # ---------------------------------------------------------
        # Adaptive temperature / effective queue configuration
        # ---------------------------------------------------------

        self.temp_queue_config = client_config.get(
            "temp_queue_config",
            {}
        )

        self.current_temperature = 0.05
        self.current_queue_size = 50


        adaptive_enabled = self.adaptive_gamma_config.get(
            "enabled",
            False
        )

        if adaptive_enabled:

            gamma_min = self.adaptive_gamma_config.get(
                "gamma_min",
                0.02
            )

            gamma_max = self.adaptive_gamma_config.get(
                "gamma_max",
                0.10
            )

            (
                self.class_counts,
                self.entropy,
                self.gamma
            ) = get_client_adaptive_gamma(
                self.data,
                num_classes=21,
                gamma_min=gamma_min,
                gamma_max=gamma_max
            )

        else:

            self.class_counts = None
            self.entropy = None

            self.gamma = self.adaptive_gamma_config.get(
                "baseline_gamma",
                0.05
            )

        print(
            f"[Client {self.id}] "
            f"Entropy={self.entropy}, "
            f"Gamma={self.gamma}, "
            f"LogitAdjustTau={self.logit_adjust_tau}",
            flush=True
        )



    # ROUND6_DIAGNOSTIC
    def _debug_log(self, message):
        try:
            path = Path("round6_debug.log")
            with open(path, "a", buffering=1) as f:
                f.write(str(message) + "\n")
                f.flush()
        except Exception:
            pass

    def get_temp_queue_schedule(self, round_number):
        """
        Round-dependent temperature and effective queue schedule.

        Physical MoCo queue remains K=50.
        CCLoss uses only the active queue_size.
        """

        cfg = self.temp_queue_config

        if not cfg.get("enabled", False):
            return 0.05, 50

        start_temp = float(
            cfg.get("start_temperature", 0.10)
        )

        end_temp = float(
            cfg.get("end_temperature", 0.05)
        )

        start_queue = int(
            cfg.get("start_queue", 10)
        )

        end_queue = int(
            cfg.get("end_queue", 50)
        )

        total_rounds = max(
            int(cfg.get("total_rounds", 10)),
            2
        )

        progress = min(
            max(
                (float(round_number) - 1.0)
                / (float(total_rounds) - 1.0),
                0.0
            ),
            1.0
        )

        temperature = (
            start_temp
            + progress * (end_temp - start_temp)
        )

        queue_size = int(
            round(
                start_queue
                + progress * (end_queue - start_queue)
            )
        )

        queue_size = max(
            1,
            min(queue_size, 50)
        )

        return temperature, queue_size

    def client_update_cl(self, round_number=1):

        self._debug_log(
            f"[DEBUG] Client {self.id} ENTER client_update_cl "
            f"round={round_number}"
        )
        """Update local model using local dataset."""
    
        # Move only the currently active client's model to GPU
        self.model = self.model.to(self.device)
        self.model.train()

        # ---------------------------------------------------------
        # Adaptive temperature and queue
        # ---------------------------------------------------------

        temperature, queue_size = (
            self.get_temp_queue_schedule(round_number)
        )

        self.current_temperature = temperature
        self.current_queue_size = queue_size

        print(
            f"Client {self.id}: "
            f"Round={round_number}, "
            f"Temperature={temperature:.4f}, "
            f"Queue={queue_size}",
            flush=True
        )


        optimizer = eval(self.optimizer)(
            self.model.parameters(),
            **self.optim_config
        )
    
        print(
            f"Client {self.id}: entered client_update_cl",
            flush=True
        )
    
        for e in range(self.local_epoch):
    
            print(
                f"Client {self.id}: epoch {e} start",
                flush=True
            )
    
            print(
                f"Client {self.id}: entering dataloader",
                flush=True
            )
    
            self._debug_log(
                f"[DEBUG] Client {self.id} BEFORE dataloader "
                f"round={round_number}"
            )

            for batch_idx, (data, labels) in enumerate(self.dataloader):

                self._debug_log(
                    f"[DEBUG] Client {self.id} BATCH_START "
                    f"round={round_number} batch={batch_idx}"
                )
    
                print(
                    f"Client {self.id}: batch {batch_idx} loaded",
                    flush=True
                )
    
                data[0] = data[0].float().to(self.device)
                data[1] = data[1].float().to(self.device)
    
                labels = labels.long().to(self.device)
    
                print(
                    f"Client {self.id}: before model",
                    flush=True
                )
    
                self._debug_log(
                    f"[DEBUG] Client {self.id} BEFORE model "
                    f"round={round_number} batch={batch_idx}"
                )

                features, labels, logits = self.model(
                    im_q=data[0],
                    im_k=data[1],
                    labels=labels
                )

                self._debug_log(
                    f"[DEBUG] Client {self.id} AFTER model "
                    f"round={round_number} batch={batch_idx}"
                )
    
                print(
                    f"Client {self.id}: after model",
                    flush=True
                )
    
                if self.loss_config["loss"] == "ccloss":

                    self._debug_log(
                        f"[DEBUG] Client {self.id} BEFORE CCLoss "
                        f"round={round_number} batch={batch_idx}"
                    )

                    loss = self.criterion_cl(
                        features,
                        labels,
                        logits,
                        gamma=self.gamma,
                        temperature=temperature,
                        queue_size=queue_size,
                        class_counts=self.class_counts,
                        logit_adjust_tau=self.logit_adjust_tau
                    )

                    self._debug_log(
                        f"[DEBUG] Client {self.id} AFTER CCLoss "
                        f"round={round_number} batch={batch_idx}"
                    )

                elif self.loss_config["loss"] == "focal":
                
                    loss = self.focal_loss(
                        logits,
                        labels[:logits.size(0)]
                    )
    
                print(
                    f"Client {self.id}: after loss",
                    flush=True
                )
    
                optimizer.zero_grad()
    
                self._debug_log(
                    f"[DEBUG] Client {self.id} BEFORE backward "
                    f"round={round_number} batch={batch_idx}"
                )

                loss.backward()

                self._debug_log(
                    f"[DEBUG] Client {self.id} AFTER backward "
                    f"round={round_number} batch={batch_idx}"
                )
    
                print(
                    f"Client {self.id}: after backward",
                    flush=True
                )
    
                self._debug_log(
                    f"[DEBUG] Client {self.id} BEFORE optimizer.step "
                    f"round={round_number} batch={batch_idx}"
                )

                optimizer.step()

                self._debug_log(
                    f"[DEBUG] Client {self.id} AFTER optimizer.step "
                    f"round={round_number} batch={batch_idx}"
                )

    
                print(
                    f"Client {self.id}: after step",
                    flush=True
                )
    
                # Release batch tensors and their computation graphs
                del loss, features, logits

                # data is a list containing the two augmented batches
                del data, labels

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
        # ---------------------------------------------------------
        # Local training complete: move model back to CPU
        # ---------------------------------------------------------
        self.model = self.model.cpu()

        # Optimizer may hold references/state on GPU
        del optimizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


