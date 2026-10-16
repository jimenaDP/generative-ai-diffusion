import torch
import numpy as np


class NoiseSchedule:
    def __init__(
        self,
        schedule=None,
        sigma=25.0,
        beta_min=None,
        beta_max=None,
        s=0.008,
    ):
        self.schedule = schedule
        self.sigma = sigma
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.s = s

    def _s_tensor(self, t):
        return torch.tensor(self.s, device=t.device, dtype=t.dtype)

    # -----------------
    # VE schedules
    # -----------------

    def diffusion_t_ve(self, t):
        """
        VE exponential diffusion coefficient.

        Forward SDE:
            dx(t) = g(t) dW(t)

        with:
            g(t) = sigma^t
        """
        return torch.pow(self.sigma, t)

    def sigma_t_ve(self, t):
        """
        Marginal standard deviation for VE exponential noise.

        Since:
            sigma_t^2 = integral_0^t g(s)^2 ds

        and:
            g(t) = sigma^t

        then:
            sigma_t^2 = (sigma^(2t) - 1) / (2 log(sigma))
        """
        return torch.sqrt(
            0.5 * (torch.pow(self.sigma, 2.0 * t) - 1.0) / np.log(self.sigma)
        )

    # -----------------
    # VP schedules
    # -----------------

    def alpha_bar_vp_cosine(self, t):
        s_tensor = self._s_tensor(t)

        a_bar = (
            torch.cos((t + s_tensor) / (1 + s_tensor) * torch.pi / 2) ** 2
            / torch.cos((s_tensor / (1 + s_tensor)) * torch.pi / 2) ** 2
        )

        return torch.clamp(a_bar, min=1.0e-12, max=1.0)

    def beta_t_vp(self, t):
        if self.schedule == "linear":
            return self.beta_min + t * (self.beta_max - self.beta_min)

        elif self.schedule == "cosine":
            beta = (torch.pi / (1 + self.s)) * torch.tan(
                (t + self.s) / (1 + self.s) * torch.pi / 2
            )
            return torch.clamp(beta, min=1.0e-12, max=50.0)

        else:
            raise NotImplementedError(
                f"Schedule '{self.schedule}' no soportado para VP."
            )

    def mu_t_vp(self, x_0, t):
        if self.schedule == "linear":
            integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
            coeff = torch.exp(-0.5 * integral)
            return coeff[:, None, None, None] * x_0

        elif self.schedule == "cosine":
            a_bar = self.alpha_bar_vp_cosine(t)
            return torch.sqrt(a_bar)[:, None, None, None] * x_0

        else:
            raise NotImplementedError(
                f"Schedule '{self.schedule}' no soportado para VP."
            )

    def sigma_t_vp(self, t):
        if self.schedule == "linear":
            integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
            return torch.sqrt(1 - torch.exp(-integral))

        elif self.schedule == "cosine":
            a_bar = self.alpha_bar_vp_cosine(t)
            return torch.sqrt(torch.clamp(1.0 - a_bar, min=1.0e-12))

        else:
            raise NotImplementedError(
                f"Schedule '{self.schedule}' no soportado para VP."
            )