import torch

def conditional_loss_function(diffusion_process, score_model, x_0, y, eps=1e-5):
    t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps
    z = torch.randn_like(x_0)

    sigma = diffusion_process.sigma_t(t)
    x_t = diffusion_process.mu_t(x_0, t) + sigma[:, None, None, None] * z

    score = score_model(x_t, t, y)

    loss = torch.mean(
        torch.sum(
            (score * sigma[:, None, None, None] + z) ** 2,
            dim=(1, 2, 3)
        )
    )
    return loss