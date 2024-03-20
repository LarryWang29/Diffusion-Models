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
    # Clip the alpha values
    alpha_t = torch.clip(alpha_t, 1e-7, 0.9999)
    beta_t = 1 - alpha_t[1:] / alpha_t[:-1]

    # Add in 0 for the first beta
    beta_t = torch.cat([torch.tensor([0.0]), beta_t])
    # Clip the beta values
    beta_t = torch.clip(beta_t, 0.0001, 0.1)
    return {"beta_t": beta_t,
            "alpha_t": alpha_t}


def inverse_linear_schedule(beta1=0.0001, beta2=0.02, T=1000):
    "Returns an inverse linear schedule."
    beta_t = (beta2 - beta1) * \
        torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    # Reverse the schedule
    beta_t = torch.flip(beta_t, [0])

    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))
    # Cumprod in log-space (better precision)
    return {"beta_t": beta_t, "alpha_t": alpha_t}


def constant_noise_schedule(T=1000):
    """Returns a constant noise schedule."""
    beta_t = torch.ones(T+1) / 100
    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))

    return {"beta_t": beta_t, "alpha_t": alpha_t}


# cosine_beta_schedule()
# constant_noise_schedule()
# ddpm_schedules()
# # Plot the sine noise schedule
# import matplotlib.pyplot as plt
# import numpy as np

# cosine_schedule = cosine_beta_schedule()
# plt.plot(cosine_schedule["beta_t"].numpy())

# # Plot the linear schedule
# linear_schedule = ddpm_schedules()
# plt.plot(linear_schedule["beta_t"].numpy())

# # Plot the constant schedule
# constant_schedule = constant_noise_schedule()
# plt.plot(constant_schedule["beta_t"].numpy())

# # Plot the reciprocal schedule
# reciprocal_schedule = inverse_linear_schedule()
# plt.plot(reciprocal_schedule["beta_t"].numpy())

# plt.title("Linear Noise Schedule")
# plt.xlabel("Time")
# plt.ylabel("Beta")
# plt.show()
