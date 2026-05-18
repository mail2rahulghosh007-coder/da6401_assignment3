"""
Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           https://arxiv.org/abs/1706.03762

Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


# ─────────────────────────────────────────────────────────────────────────────
# TODO: Implement the NoamScheduler class below
# ─────────────────────────────────────────────────────────────────────────────

class NoamScheduler(LRScheduler):
    """
    Noam learning rate scheduler as described in "Attention Is All You Need".

    Applies a warm-up phase where LR increases linearly, followed by
    inverse-square-root decay.

    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Args:
        optimizer    : wrapped optimizer
        d_model      : model dimensionality
        warmup_steps : number of warm-up steps
        last_epoch   : index of the last epoch (-1 to start fresh)
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        # step counter starts at 0 in LRScheduler; we compute step as _step_count
        super().__init__(optimizer, last_epoch=last_epoch)

    def _compute_lr(self, step):
        """Closed-form Noam formula (step is 1-based)."""
        step = max(step, 1)
        return (self.d_model ** -0.5) * min(step ** -0.5,
                                             step * (self.warmup_steps ** -1.5))

    def get_lr(self):
        # _step_count is 1 after the first .step() call
        step = self._step_count
        lr   = self._compute_lr(step)
        return [lr for _ in self.base_lrs]
