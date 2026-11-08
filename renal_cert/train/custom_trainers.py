"""Seeded nnU-Net v2 trainer subclasses supporting custom epoch schedules (100 epochs).

Cites:
    PDF §5.2 (nnU-Net v2 training configuration and residual-encoder presets)
    PDF §6.2 (100-epoch schedule; seeded trainer subclasses)
"""

import os
import random
import numpy as np

# We provide fallback dummy inheritance if nnunetv2 is not installed locally on Windows
try:
    import torch
    import pathlib
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    HAS_NNUNET = True

    # Self-healing patch for PyTorch 2.4+ (where deprecated 'verbose=False' was removed from LRScheduler)
    try:
        import nnunetv2.training.lr_scheduler.polylr as plr
        _plr_file = pathlib.Path(plr.__file__)
        _content = _plr_file.read_text()
        if ", False)" in _content:
            _plr_file.write_text(_content.replace(", False)", ")"))
    except Exception:
        pass

    class PolyLRSchedulerCompatible(torch.optim.lr_scheduler._LRScheduler):
        """PolyLRScheduler compatible with all PyTorch releases (avoids removed verbose parameter)."""
        def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
            self.optimizer = optimizer
            self.initial_lr = initial_lr
            self.max_steps = max_steps
            self.exponent = exponent
            self.ctr = 0
            super().__init__(optimizer, current_step if current_step is not None else -1)

        def step(self, current_step=None):
            if current_step is None or current_step == -1:
                current_step = self.ctr
                self.ctr += 1
            new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

except ImportError:
    class nnUNetTrainer:
        num_epochs = 1000
        def initialize(self): pass
    PolyLRSchedulerCompatible = None
    HAS_NNUNET = False


class nnUNetTrainer_KiTS23_100ep(nnUNetTrainer):
    """Base 100-epoch trainer for KiTS23 study (exact user constraint: 100 epochs; unpack_dataset=False to save ~35GB disk)."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = False,
                 device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset=False, device=device)
        self.num_epochs = 100  # User instruction: all training epoch are in the number of 100 epoch

    def configure_optimizers(self):
        if HAS_NNUNET:
            optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = PolyLRSchedulerCompatible(optimizer, self.initial_lr, self.num_epochs)
            return optimizer, lr_scheduler
        return super().configure_optimizers()


class nnUNetTrainer_KiTS23_Smoke(nnUNetTrainer):
    """Ultra-fast 2-epoch sanity trainer for smoke tier pipeline verification."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = False,
                 device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset=False, device=device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = 5

    def configure_optimizers(self):
        if HAS_NNUNET:
            optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = PolyLRSchedulerCompatible(optimizer, self.initial_lr, self.num_epochs)
            return optimizer, lr_scheduler
        return super().configure_optimizers()


# Helper to generate seeded subclasses
def create_seeded_trainer_class(base_cls, seed: int, class_name: str):
    def initialize(self):
        super(base_cls, self).initialize()
        if HAS_NNUNET:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    attrs = {"initialize": initialize}
    return type(class_name, (base_cls,), attrs)


# Pre-register seeded 100-epoch trainers for core pool
nnUNetTrainer_KiTS23_100ep_s101 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_100ep, 101, "nnUNetTrainer_KiTS23_100ep_s101")
nnUNetTrainer_KiTS23_100ep_s102 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_100ep, 102, "nnUNetTrainer_KiTS23_100ep_s102")
nnUNetTrainer_KiTS23_100ep_s103 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_100ep, 103, "nnUNetTrainer_KiTS23_100ep_s103")
nnUNetTrainer_KiTS23_100ep_s104 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_100ep, 104, "nnUNetTrainer_KiTS23_100ep_s104")
nnUNetTrainer_KiTS23_100ep_s105 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_100ep, 105, "nnUNetTrainer_KiTS23_100ep_s105")

nnUNetTrainer_KiTS23_100ep_s201 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_100ep, 201, "nnUNetTrainer_KiTS23_100ep_s201")
nnUNetTrainer_KiTS23_100ep_s202 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_100ep, 202, "nnUNetTrainer_KiTS23_100ep_s202")
nnUNetTrainer_KiTS23_100ep_s203 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_100ep, 203, "nnUNetTrainer_KiTS23_100ep_s203")

nnUNetTrainer_KiTS23_100ep_s301 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_100ep, 301, "nnUNetTrainer_KiTS23_100ep_s301")
nnUNetTrainer_KiTS23_100ep_s302 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_100ep, 302, "nnUNetTrainer_KiTS23_100ep_s302")

# Smoke trainers
nnUNetTrainer_KiTS23_Smoke_s101 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_Smoke, 101, "nnUNetTrainer_KiTS23_Smoke_s101")
nnUNetTrainer_KiTS23_Smoke_s201 = create_seeded_trainer_class(nnUNetTrainer_KiTS23_Smoke, 201, "nnUNetTrainer_KiTS23_Smoke_s201")
