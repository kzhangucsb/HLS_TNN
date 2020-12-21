#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:28:55 2020

@author: zkq
"""

import numpy as np


def adam_step(param, grad, step, buffer, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    if buffer:
        exp_avg = buffer[0]
        exp_avg_sq = buffer[1]
        

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    np.copyto(exp_avg, exp_avg * beta1 + grad * (1-beta1))
    np.copyto(exp_avg_sq, exp_avg_sq * beta1 + grad * grad * (1-beta1))
    
    denom = (np.sqrt(exp_avg_sq) / np.sqrt(bias_correction2)) + eps

    step_size = lr / bias_correction1

    np.copyto(param, param - lr * exp_avg / denom)
