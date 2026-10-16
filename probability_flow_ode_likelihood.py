import math
import torch


def gaussian_log_prob(x, sigma):
    """
    Compute log p(x) for a diagonal Gaussian N(0, sigma^2 I).

    Args:
        x: Tensor of shape (B, C, H, W).
        sigma: Tensor of shape (B,) or scalar.

    Returns:
        logp: Tensor of shape (B,).
    """
    B = x.shape[0]
    D = x[0].numel()

    if not torch.is_tensor(sigma):
        sigma = torch.full((B,), float(sigma), device=x.device, dtype=x.dtype)
    else:
        sigma = sigma.to(device=x.device, dtype=x.dtype).reshape(B)

    x_flat = x.reshape(B, -1)

    logp = -0.5 * (
        D * torch.log(2.0 * torch.pi * sigma**2)
        + (x_flat**2).sum(dim=1) / sigma**2
    )

    return logp

def divergence_hutchinson(drift_fn, x, t):
    """
    Estimate div_x drift_fn(x,t) using Hutchinson's estimator.
    """
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)

        e = torch.randint_like(x, low=0, high=2).float()
        e = 2.0 * e - 1.0

        drift = drift_fn(x, t)

        inner_product = torch.sum(drift * e)

        grad = torch.autograd.grad(
            inner_product,
            x,
            create_graph=False,
            retain_graph=False,
        )[0]

        divergence = torch.sum(grad * e, dim=(1, 2, 3))

    return divergence, drift.detach()


def probability_flow_log_likelihood(
    x_0,
    t_0,
    t_end,
    n_steps,
    ode_drift_coefficient,
    prior_logp_fn,
):
    device = x_0.device
    B = x_0.shape[0]
    D = x_0[0].numel()

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x = x_0.detach()
    logp_correction = torch.zeros(B, device=device, dtype=x_0.dtype)

    for n in range(n_steps):
        t = torch.full(
            (B,),
            times[n].item(),
            device=device,
            dtype=x_0.dtype,
        )

        divergence, drift = divergence_hutchinson(
            ode_drift_coefficient,
            x,
            t,
        )

        x = (x + drift * dt).detach()
        logp_correction = logp_correction + divergence.detach() * dt

    logp_T = prior_logp_fn(x)
    logp_0 = logp_T + logp_correction
    bpd_continuous = -logp_0 / (D * math.log(2.0))
    bpd = bpd_continuous + 8.0

    return logp_0, bpd, logp_T