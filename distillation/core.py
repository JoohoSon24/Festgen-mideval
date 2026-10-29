"""DDIM targets based on Salimans & Ho (2022), Algorithms 2 and Appendix D.

Original implementation adapted to the repository's discrete linear VP schedule.
"""
from __future__ import annotations
import torch
from teacher.cfg import apply_cfg


def coefficients(alpha_bar: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    # -1 represents the exact clean endpoint, with alpha=1, sigma=0.
    a=alpha_bar[times.clamp_min(0)]
    a=torch.where(times<0,torch.ones_like(a),a)
    return a.sqrt()[:,None,None,None], (1-a).sqrt()[:,None,None,None]


def ddim_step(x: torch.Tensor, x0: torch.Tensor, a: torch.Tensor, s: torch.Tensor,
              an: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
    return an*x0+sn*(x-a*x0)/s


def velocity_x0(network, x: torch.Tensor, times: torch.Tensor,
                labels: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    a,s=coefficients(alpha_bar,times)
    return a*x-s*network(x,times,labels).float()


def guided_x0(network,x,times,labels,alpha_bar,scale: float) -> torch.Tensor:
    # CFG arithmetic remains defined only in teacher.cfg.apply_cfg.
    uncond=network(x,times,torch.full_like(labels,network.num_classes))
    cond=network(x,times,labels)
    eps=apply_cfg(uncond,cond,scale)
    a,s=coefficients(alpha_bar,times)
    return ((x-s*eps)/a).clamp(-1,1)


def merged_target(x: torch.Tensor, end: torch.Tensor, a: torch.Tensor,s: torch.Tensor,
                  an: torch.Tensor,sn: torch.Tensor) -> torch.Tensor:
    ratio=sn/s
    # Endpoint inversion must not clamp the effective clean prediction: a merged
    # target may lie outside [-1,1] even when individual teacher predictions clip.
    return (end-ratio*x)/(an-ratio*a)


def target_velocity(x,x0,a,s):
    return (a*x-x0)/s


def velocity_loss(prediction,target,a,s):
    # max(SNR,1)*MSE(x0) = max(alpha^2,sigma^2)*MSE(v).
    weight=torch.maximum(a.square(),s.square())
    return (weight*(prediction.float()-target.float()).square()).mean()


@torch.no_grad()
def sample_velocity(network,alpha_bar,times,labels,seed:int,clip_x0:bool=False):
    generator=torch.Generator(device=labels.device).manual_seed(seed)
    x=torch.randn((len(labels),3,64,64),device=labels.device,generator=generator)
    for i,time in enumerate(times.tolist()):
        t=torch.full_like(labels,time)
        next_t=torch.full_like(labels,times[i+1].item() if i+1<len(times) else -1)
        a,s=coefficients(alpha_bar,t);an,sn=coefficients(alpha_bar,next_t)
        x0=velocity_x0(network,x,t,labels,alpha_bar)
        if clip_x0: x0=x0.clamp(-1,1)
        x=ddim_step(x,x0,a,s,an,sn)
    return x.clamp(-1,1)
