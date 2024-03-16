import torch


def ddpm_schedules(beta1=0.0001, beta2=0.02, T=1000):
    """Returns pre-computed schedules for DDPM sampling with a linear noise
    schedule."""
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * \
        torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))
    # Cumprod in log-space (better precision)

    return {"beta_t": beta_t, "alpha_t": alpha_t}


def cosine_beta_schedule(s=0.008, T=1000):
    """Returns a cosine beta schedule."""
    f_t = torch.cos((torch.pi / 2) *
                    (torch.linspace(0, T, T+1)/T + s) / (1 + s)).pow(2)
    alpha_t = f_t / f_t[0]
    beta_t = 1 - alpha_t[1:] / alpha_t[:-1]
    return {"beta_t": torch.clip(beta_t, 0.0001, 0.05),
            "alpha_t": torch.clip(alpha_t, 0.001, 0.999)}
