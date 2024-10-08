import numpy as np


def cosine_lr(base_lr, decay_steps, total_steps):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))


def poly_lr(base_lr, decay_steps, total_steps, end_lr=0.0001, power=0.9):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield (base_lr - end_lr) * ((1.0 - step_ / decay_steps) ** power) + end_lr


def exponential_lr(base_lr, decay_steps, decay_rate, total_steps, staircase=False):
    for i in range(total_steps):
        if staircase:
            power_ = i // decay_steps
        else:
            power_ = float(i) / decay_steps
        yield base_lr * (decay_rate ** power_)

def warmup_poly_lr(warmup_lr, base_lr, warmup_steps, decay_steps, total_steps, end_lr=0.0001, power=0.9):
    for i in range(total_steps):
        if(i <= warmup_steps):
            yield (base_lr - warmup_lr) * (i / warmup_steps) + warmup_lr
        else:
            step_ = min(i, decay_steps)
            yield (base_lr - end_lr) * ((1.0 - (step_ - warmup_steps) / (decay_steps - warmup_steps)) ** power) + end_lr