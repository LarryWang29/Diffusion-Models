"""!@file src/schedulers.py

@brief Schedulers used for diffusion models.

@details This file contains the implementation of the schedulers used for
diffusion models. The schedulers are used to schedule the noise values for the
diffusion process. The schedulers implemented in this file are the linear (DDPM)
scheduler, cosine scheduler, inverse linear scheduler, and constant noise
scheduler.

@author Larry Wang
@Date 22/03/2024
"""
import torch


def ddpm_schedules(beta1=0.0001, beta2=0.02, T=1000):
    """
    !@brief Returns the beta and alpha values for the DDPM scheduler.

    @details This function returns the beta and alpha values for the DDPM
    scheduler. The beta and alpha values are used to calculate the diffusion
    process. The noise is linearly spaced between the beta1 and beta2 values.

    @param beta1 The initial beta value for the scheduler, default is 0.0001.
    @param beta2 The final beta value for the scheduler, default is 0.02.
    @param T The total number of diffusion time steps for the scheduler.

    @return A dictionary containing the beta and alpha values for the scheduler.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * \
        torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))
    # Cumprod in log-space (better precision)
    return {"beta_t": beta_t, "alpha_t": alpha_t}


def cosine_beta_schedule(s=0.008, T=1000):
    """
    !@brief Returns the beta and alpha values for the cosine scheduler.

    @details This function returns the beta and alpha values for the cosine
    scheduler. The beta and alpha values are used to calculate the diffusion
    process. The implementation is based on the cosine scheduler used in the
    paper Improved Denoising Diffusion Probabilistic Models.

    @param s An offset factor for the cosine scheduler, default is 0.008.
    @param T The total number of diffusion time steps for the scheduler.

    @return A dictionary containing the beta and alpha values for the scheduler.
    """
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
    """
    !@brief Returns the beta and alpha values for the inverse linear scheduler.

    @details This function returns the beta and alpha values for the inverse
    linear scheduler. The beta and alpha values are used to calculate the
    diffusion process. The inverse linear scheduler is the reverse of the linear
    scheduler.

    @param beta1 The initial beta value for the scheduler, default is 0.0001.
    @param beta2 The final beta value for the scheduler, default is 0.02.
    @param T The total number of diffusion time steps for the scheduler.

    @return A dictionary containing the beta and alpha values for the scheduler.
    """
    beta_t = (beta2 - beta1) * \
        torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    # Reverse the schedule
    beta_t = torch.flip(beta_t, [0])

    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))
    # Cumprod in log-space (better precision)
    return {"beta_t": beta_t, "alpha_t": alpha_t}


def constant_noise_schedule(T=1000):
    """
    !@brief Returns the beta and alpha values for the constant noise scheduler.

    @details This function returns the beta and alpha values for the constant
    noise scheduler. In this schedule beta is constant and equal to 1/100.

    @param T The total number of diffusion time steps for the scheduler.
    
    @return A dictionary containing the beta and alpha values for the scheduler.
    """
    beta_t = torch.ones(T+1) / 100
    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))

    return {"beta_t": beta_t, "alpha_t": alpha_t}
