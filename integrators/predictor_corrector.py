# -*- coding: utf-8 -*-
"""
Predictor-Corrector integrator for score-based SDEs.
"""

import torch
from torch import Tensor
from typing import Callable, Tuple


def predictor_corrector_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    backward_drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    score_model: Callable[[Tensor, Tensor], Tensor],
    corrector_steps: int = 1,
    corrector_step_size: float = 0.01,
    normalize_score: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Predictor-Corrector sampler for score-based diffusion models.

    Args:
        x_0: Initial noise, shape (B, C, H, W)
        t_0: Initial time
        t_end: Final time
        n_steps: Number of time steps
        backward_drift_coefficient: reverse drift f_rev(x,t)
        diffusion_coefficient: diffusion g(t)
        score_model: score network s_theta(x,t)
        corrector_steps: number of Langevin steps per time step
        corrector_step_size: step size for Langevin corrector
        normalize_score: whether to normalize score in corrector

    Returns:
        times: time grid
        x_t: trajectories, shape (B, C, H, W, n_steps + 1)
    """
    device = x_0.device
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]
    dt_abs = torch.abs(dt)

    x_t = torch.empty((*x_0.shape, n_steps + 1), device=device)
    x_t[..., 0] = x_0
    x = x_0

    for n in range(n_steps):
        t = torch.full((x.shape[0],), times[n], device=device, dtype=x.dtype)

        # -------------------------
        # Corrector: Langevin steps
        # -------------------------
        for _ in range(corrector_steps):
            score = score_model(x, t)

            if normalize_score:
                norm = torch.norm(
                    score.reshape(score.shape[0], -1),
                    dim=1
                ).view(-1, 1, 1, 1) + 1e-12
                score = score / norm

            noise = torch.randn_like(x)
            eps = torch.tensor(corrector_step_size, device=device, dtype=x.dtype)

            x = x + eps * score + torch.sqrt(2.0 * eps) * noise

        # -------------------------
        # Predictor: reverse SDE step
        # -------------------------
        noise = torch.randn_like(x)
        drift = backward_drift_coefficient(x, t)
        g_t = diffusion_coefficient(t).view(-1, 1, 1, 1)

        x = x + drift * dt + g_t * torch.sqrt(dt_abs) * noise
        x_t[..., n + 1] = x

    return times, x_t