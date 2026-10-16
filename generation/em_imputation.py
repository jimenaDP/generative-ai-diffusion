from __future__ import annotations
from typing import Callable, Union, Tuple

import numpy as np
import torch
from torch import Tensor

def right_half_mask(x):
    """
    x: (B, C, H, W)
    Returns mask with 1 in observed region, 0 in missing region.
    """
    mask = torch.ones_like(x)
    _, _, H, W = x.shape
    mask[:, :, :, W // 2:] = 0.0
    return mask

def euler_maruyama_imputation(
    x_0: Tensor,
    x_obs: Tensor,
    mask: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    mu_t: Callable[[Tensor, Tensor], Tensor],
    sigma_t: Callable[[Tensor], Tensor],
    seed: Union[int, None] = None
) -> Tensor:
    """Euler-Maruyama integrator for imputation (approximate)

    This function integrates the reverse SDE while enforcing consistency with
    the observed part of an image. At each step, the sample is first updated
    with Euler-Maruyama and then combined with the observed image using a mask.

    Args:
        x_0: Initial images of dimensions
            (batch_size, n_channels, image_height, image_width).
            In reverse diffusion, this is typically the noisy sample at time t_0.
        x_obs: Observed images to be imputed, with the same shape as x_0.
        mask: Binary mask with the same shape as x_obs:
            - 1 for observed pixels
            - 0 for missing pixels
        t_0: Initial time of the integration interval.
        t_end: Endpoint of the integration interval.
        n_steps: Number of integration steps.
        drift_coefficient: Function of :math:`(x(t), t)` that defines the drift term.
        diffusion_coefficient: Function of :math:`(t)` that defines the diffusion term.
        mu_t: Function of :math:`(x_0, t)` giving the mean of the forward process.
        sigma_t: Function of :math:`(t)` giving the standard deviation of the
            forward process.
        seed: Seed for the random number generator.

    Returns:
        x_t: Trajectories resulting from the integration of the SDE.
             The shape is (*np.shape(x_0), (n_steps + 1)).

    Notes:
        The implementation is fully vectorized except for a loop over time.
        After each reverse step, the observed region is replaced by a noisy
        version of x_obs at the same time step.

    Examples:
        >>> times, x_t = euler_maruyama_imputation(
        ...     x_0=image_T,
        ...     x_obs=x_obs,
        ...     mask=mask,
        ...     t_0=1.0,
        ...     t_end=1.0e-3,
        ...     n_steps=500,
        ...     drift_coefficient=drift_backward,
        ...     diffusion_coefficient=diffusion_model.diffusion,
        ...     mu_t=diffusion_model.mu_t,
        ...     sigma_t=diffusion_model.sigma_t,
        ... )
        >>> print(np.shape(x_t))
        torch.Size([batch_size, n_channels, image_height, image_width, 501])
    """

    # Fix random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    device = x_0.device

    # Create the time grid
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    # Store all trajectories
    x_t = torch.tensor(
        np.empty((*np.shape(x_0), len(times))),
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_0

    # Gaussian noise for the reverse SDE
    z = torch.randn_like(x_t)
    z[..., -1] = 0.0  # No noise injection in the last step

    # Gaussian noise for the noisy observed image
    z_obs = torch.randn_like(x_t)
    z_obs[..., -1] = 0.0

    for n, t in enumerate(times[:-1]):
        # Current time and next time as batch-shaped tensors
        t = torch.ones(x_0.shape[0], device=device) * t
        t_next = torch.ones(x_0.shape[0], device=device) * times[n + 1]

        # Euler-Maruyama reverse step
        x_t[..., n + 1] = (
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t) * dt
            + diffusion_coefficient(t).view(-1, 1, 1, 1)
              * np.sqrt(np.abs(dt))
              * z[..., n]
        )

        # Noisy version of the observed image at the new time step
        sigma = sigma_t(t_next)
        x_obs_t = (
            mu_t(x_obs, t_next)
            + sigma.view(-1, 1, 1, 1) * z_obs[..., n + 1]
        )

        # Keep observed pixels from x_obs_t and generate only the missing ones
        x_t[..., n + 1] = (
            mask * x_obs_t
            + (1.0 - mask) * x_t[..., n + 1]
        )

    return times, x_t

def predictor_corrector_imputation(
    x_0: Tensor,
    x_obs: Tensor,
    mask: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    backward_drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    score_model: Callable[[Tensor, Tensor], Tensor],
    mu_t: Callable[[Tensor, Tensor], Tensor],
    sigma_t: Callable[[Tensor], Tensor],
    corrector_steps: int = 1,
    corrector_step_size: float = 0.01,
    normalize_score: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Predictor-Corrector sampler adapted for image imputation.

    This function integrates the reverse SDE using a Predictor-Corrector method
    while enforcing consistency with the observed part of an image. At each time
    step, the method first applies Langevin corrector steps, then a reverse SDE
    predictor step, and finally replaces the observed pixels using a noisy
    version of the original observed image at the same time level.

    Args:
        x_0: Initial noisy images of shape
            (batch_size, n_channels, image_height, image_width).
        x_obs: Observed images to be imputed, with the same shape as x_0.
        mask: Binary mask with the same shape as x_obs.
            mask = 1 for observed pixels and mask = 0 for missing pixels.
        t_0: Initial time of the reverse integration interval, typically T.
        t_end: Final time of the reverse integration interval, typically 1e-3.
        n_steps: Number of time steps.
        backward_drift_coefficient: Function of (x_t, t) defining the reverse
            drift coefficient.
        diffusion_coefficient: Function of t defining the diffusion coefficient
            g(t).
        score_model: Score network s_theta(x_t, t).
        mu_t: Function giving the forward-process mean mu_t(x_obs, t).
        sigma_t: Function giving the forward-process standard deviation sigma_t(t).
        corrector_steps: Number of Langevin corrector steps per time step.
        corrector_step_size: Step size for the Langevin corrector.
        normalize_score: Whether to normalize the score in the corrector step.

    Returns:
        times: 1D tensor containing the integration time grid.
        x_t: Tensor containing the full imputation trajectory, with shape
            (batch_size, n_channels, image_height, image_width, n_steps + 1).

    Notes:
        The observed region is not kept clean during the reverse process.
        Instead, it is replaced by a noisy version of x_obs at the current time,
        so that it has the same noise level as the generated sample.
    """
    device = x_0.device
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]
    dt_abs = torch.abs(dt)

    x_t = torch.empty((*x_0.shape, n_steps + 1), device=device)
    x_t[..., 0] = x_0
    x = x_0

    for n in range(n_steps):
        t = torch.full(
            (x.shape[0],),
            times[n].item(),
            device=device,
            dtype=x.dtype,
        )

        #corrector
        for _ in range(corrector_steps):
            score = score_model(x, t)

            if normalize_score:
                norm = torch.norm(
                    score.reshape(score.shape[0], -1),
                    dim=1,
                ).view(-1, 1, 1, 1) + 1e-12
                score = score / norm

            noise = torch.randn_like(x)
            eps = torch.tensor(corrector_step_size, device=device, dtype=x.dtype)

            x = x + eps * score + torch.sqrt(2.0 * eps) * noise

           
            z_obs = torch.randn_like(x_obs)
            sigma = sigma_t(t)
            x_obs_t = mu_t(x_obs, t) + sigma[:, None, None, None] * z_obs
            x = mask * x_obs_t + (1.0 - mask) * x

       #reverse SDE step
        noise = torch.randn_like(x)
        drift = backward_drift_coefficient(x, t)
        g_t = diffusion_coefficient(t).view(-1, 1, 1, 1)

        x = x + drift * dt + g_t * torch.sqrt(dt_abs) * noise

        #enforce observed pixels after predictor step
        t_next = torch.full(
            (x.shape[0],),
            times[n + 1].item(),
            device=device,
            dtype=x.dtype,
        )

        z_obs = torch.randn_like(x_obs)
        sigma = sigma_t(t_next)
        x_obs_t = mu_t(x_obs, t_next) + sigma[:, None, None, None] * z_obs

        x = mask * x_obs_t + (1.0 - mask) * x

        x_t[..., n + 1] = x

    return times, x_t

def probability_flow_ode_imputation(
    x_0: Tensor,
    x_obs: Tensor,
    mask: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    ode_drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    mu_t: Callable[[Tensor, Tensor], Tensor],
    sigma_t: Callable[[Tensor], Tensor],
) -> Tuple[Tensor, Tensor]:
    """
    Integrate the Probability Flow ODE for image imputation.

    This function performs deterministic imputation using the Probability Flow
    ODE. At each integration step, the sample is updated with an explicit Euler
    step and then combined with the observed image using a binary mask.

    Args:
        x_0: Initial noisy images of shape
            (batch_size, n_channels, image_height, image_width).
        x_obs: Observed images to be imputed, with the same shape as x_0.
        mask: Binary mask with the same shape as x_obs.
            mask = 1 for observed pixels and mask = 0 for missing pixels.
        t_0: Initial time of the reverse integration interval, typically T.
        t_end: Final time of the reverse integration interval, typically 1e-3.
        n_steps: Number of integration steps.
        ode_drift_coefficient: Function of (x_t, t) returning the drift of the
            Probability Flow ODE.
        mu_t: Function giving the forward-process mean mu_t(x_obs, t).
        sigma_t: Function giving the forward-process standard deviation sigma_t(t).

    Returns:
        times: 1D tensor containing the integration time grid.
        x_t: Tensor containing the full imputation trajectory, with shape
            (batch_size, n_channels, image_height, image_width, n_steps + 1).

    Notes:
        Unlike the reverse SDE, the Probability Flow ODE does not inject random
        noise in the generated part during the update. However, for imputation,
        the observed region is still replaced by a noisy version of the original
        observed image at each time step, so that it is consistent with the
        current diffusion time.
    """
    device = x_0.device

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x_t = torch.empty((*x_0.shape, n_steps + 1), device=device)
    x_t[..., 0] = x_0

    x = x_0

    with torch.no_grad():
        for n in range(n_steps):
            t = torch.full(
                (x.shape[0],),
                times[n].item(),
                device=device,
                dtype=x.dtype,
            )

            drift = ode_drift_coefficient(x, t)

            # Explicit Euler step for the Probability Flow ODE
            x = x + drift * dt

            # Enforce observed pixels at the next time level
            t_next = torch.full(
                (x.shape[0],),
                times[n + 1].item(),
                device=device,
                dtype=x.dtype,
            )

            z_obs = torch.randn_like(x_obs)
            sigma = sigma_t(t_next)
            x_obs_t = mu_t(x_obs, t_next) + sigma[:, None, None, None] * z_obs

            x = mask * x_obs_t + (1.0 - mask) * x

            x_t[..., n + 1] = x

    return times, x_t