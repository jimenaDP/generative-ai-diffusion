import math
import torch
from tqdm.auto import tqdm


def gaussian_log_prob(x, sigma):
    """
    Log p(x) for a diagonal Gaussian N(0, sigma^2 I).

    Args:
        x: Tensor of shape (B, C, H, W)
        sigma: scalar float or tensor of shape (B,)

    Returns:
        logp: Tensor of shape (B,)
    """
    B = x.shape[0]
    D = x[0].numel()

    if not torch.is_tensor(sigma):
        sigma = torch.full((B,), float(sigma), device=x.device, dtype=x.dtype)
    else:
        sigma = sigma.to(device=x.device, dtype=x.dtype).reshape(B)

    sigma2 = sigma ** 2
    x_flat = x.reshape(B, -1)

    quad = (x_flat ** 2).sum(dim=1) / sigma2
    log_det = D * torch.log(2.0 * torch.pi * sigma2)

    return -0.5 * (quad + log_det)


def divergence_hutchinson(drift_fn, x, t, noise_type="rademacher"):
    """
    Hutchinson trace estimator of div_x drift_fn(x, t).

    Args:
        drift_fn: function (x, t) -> tensor with same shape as x
        x: Tensor (B, C, H, W)
        t: Tensor (B,)
        noise_type: "rademacher" or "gaussian"

    Returns:
        drift: drift_fn(x, t)
        div: estimated divergence, shape (B,)
    """
    x = x.detach().requires_grad_(True)

    if noise_type == "rademacher":
        eps = torch.randint_like(x, low=0, high=2, dtype=torch.int64).float()
        eps = eps * 2.0 - 1.0
    elif noise_type == "gaussian":
        eps = torch.randn_like(x)
    else:
        raise ValueError("noise_type must be 'rademacher' or 'gaussian'")

    drift = drift_fn(x, t)
    inner = (drift * eps).reshape(x.shape[0], -1).sum(dim=1)   # (B,)

    grad = torch.autograd.grad(
        outputs=inner.sum(),
        inputs=x,
        create_graph=False,
        retain_graph=False,
    )[0]

    div = (grad * eps).reshape(x.shape[0], -1).sum(dim=1)
    return drift.detach(), div.detach()


@torch.no_grad()
def get_prior_sigma_T(diffusion_model, batch_size, t_end, device, dtype):
    """
    Compute sigma(T) from the forward process.
    """
    tT = torch.full((batch_size,), t_end, device=device, dtype=dtype)
    sigma_T = diffusion_model.sigma_t(tT)
    return sigma_T


def bpd_batch_probability_flow(
    score_model,
    diffusion_model,
    x0,
    t_start=1e-5,
    t_end=1.0,
    n_steps=500,
    noise_type="rademacher",
):
    """
    Approximate BPD for one batch of real images x0 using the probability flow ODE.

    Important:
    - This evaluates the trained model.
    - It does NOT compare Euler-Maruyama vs Predictor-Corrector as samplers.
    - It needs gradients, so do not wrap this function with torch.no_grad().

    Args:
        score_model: trained score network
        diffusion_model: your DiffusionModel instance
        x0: Tensor (B, C, H, W), real images
        t_start: small epsilon > 0
        t_end: final time T
        n_steps: number of Euler steps for the ODE
        noise_type: Hutchinson estimator noise

    Returns:
        logp0: approximate log-likelihood of x0, shape (B,)
        bpd: bits per dimension, shape (B,)
    """
    score_model.eval()

    device = x0.device
    dtype = x0.dtype
    B = x0.shape[0]
    D = x0[0].numel()

    dt = (t_end - t_start) / n_steps
    x = x0.detach().clone()
    delta_logp = torch.zeros(B, device=device, dtype=dtype)

    drift_fn = diffusion_model.get_probability_flow_drift(score_model)

    for k in range(n_steps):
        t_scalar = t_start + k * dt
        t = torch.full((B,), t_scalar, device=device, dtype=dtype)

        drift, div = divergence_hutchinson(
            drift_fn=drift_fn,
            x=x,
            t=t,
            noise_type=noise_type,
        )

        # Forward integration of the probability flow ODE: x(0) -> x(T)
        x = x + drift * dt

    
        delta_logp = delta_logp + div * dt

    sigma_T = get_prior_sigma_T(
        diffusion_model=diffusion_model,
        batch_size=B,
        t_end=t_end,
        device=device,
        dtype=dtype,
    )

    logp_T = gaussian_log_prob(x, sigma_T)
    logp_0 = logp_T + delta_logp

    bpd = -logp_0 / (D * math.log(2.0)) + 8.0 #to normalize
    return logp_0, bpd


def evaluate_bpd_dataset(
    score_model,
    diffusion_model,
    data_loader,
    device,
    t_start=1e-5,
    t_end=1.0,
    n_steps=500,
    noise_type="rademacher",
    max_batches=None,
):
    """
    Evaluate mean BPD over a dataset.

    Args:
        data_loader: batches like (images, labels) or just images
        max_batches: set an integer first to test quickly

    Returns:
        mean_bpd, std_bpd
    """
    all_bpd = []

    for i, batch in enumerate(tqdm(data_loader, desc="Computing BPD")):
        if max_batches is not None and i >= max_batches:
            break

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        _, bpd = bpd_batch_probability_flow(
            score_model=score_model,
            diffusion_model=diffusion_model,
            x0=x,
            t_start=t_start,
            t_end=t_end,
            n_steps=n_steps,
            noise_type=noise_type,
        )

        all_bpd.append(bpd.cpu())

    all_bpd = torch.cat(all_bpd)
    return all_bpd.mean().item(), all_bpd.std().item()