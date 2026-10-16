import torch
import diffusion_process as dfp
from noise_schedule import NoiseSchedule


class DiffusionModel:
    def __init__(
        self,
        type,
        schedule=None,
        sigma=25.0,
        beta_min=None,
        beta_max=None,
        s=0.008,
    ):
        self.type = type
        self.sigma = sigma

        self.noise_schedule = NoiseSchedule(
            schedule=schedule,
            sigma=sigma,
            beta_min=beta_min,
            beta_max=beta_max,
            s=s,
        )

    def drift(self, x_t, t):
        if self.type == "VE":
            return torch.zeros_like(x_t) #drift = 0

        elif self.type == "VP":
            beta = self.noise_schedule.beta_t_vp(t)
            return -0.5 * beta[:, None, None, None] * x_t

        else:
            raise NotImplementedError(
                f"Process '{self.type}' not supported."
            )

    def diffusion(self, t):
        if self.type == "VE":
            return self.noise_schedule.diffusion_t_ve(t)

        elif self.type == "VP":
            beta = self.noise_schedule.beta_t_vp(t)
            return torch.sqrt(beta)

        else:
            raise NotImplementedError(
                f"Process '{self.type}' not supported."
            )

    def mu_t(self, x_0, t):
        if self.type == "VE":
            return x_0

        elif self.type == "VP":
            return self.noise_schedule.mu_t_vp(x_0, t)

        else:
            raise NotImplementedError(
                f"Process '{self.type}' not supported."
            )

    def sigma_t(self, t):
        if self.type == "VE":
            return self.noise_schedule.sigma_t_ve(t)

        elif self.type == "VP":
            return self.noise_schedule.sigma_t_vp(t)

        else:
            raise NotImplementedError(
                f"Process '{self.type}' not supported."
            )

    def backward_drift(self, score_model, x_t, t):
        score = score_model(x_t, t)
        g = self.diffusion(t)

        if self.type == "VE":
            return - (g ** 2)[:, None, None, None] * score

        elif self.type == "VP":
            beta = self.noise_schedule.beta_t_vp(t)
            return (
                -0.5 * beta[:, None, None, None] * x_t
                - (g ** 2)[:, None, None, None] * score
            )

        else:
            raise NotImplementedError(
                f"Process '{self.type}' not supported."
            )

    def get_backward_drift(self, score_model):
        return lambda x_t, t: self.backward_drift(score_model, x_t, t)
    
    def probability_flow_drift(self, score_model, x_t, t):
        score = score_model(x_t, t)
        g = self.diffusion(t)
        f = self.drift(x_t, t)

        return f - 0.5 * (g ** 2)[:, None, None, None] * score

    def get_probability_flow_drift(self, score_model):
        return lambda x_t, t: self.probability_flow_drift(score_model, x_t, t)

    # Build GaussianDiffussionProcess
    def build(self):
        return dfp.GaussianDiffussionProcess(
            drift_coefficient=self.drift,
            diffusion_coefficient=self.diffusion,
            mu_t=self.mu_t,
            sigma_t=self.sigma_t,
        )