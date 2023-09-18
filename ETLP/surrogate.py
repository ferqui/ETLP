from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn


class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return (input_ > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (10 * torch.abs(input_) + 1.0) ** 2


class SmoothStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return (input_ > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ <= -0.5] = 0
        grad_input[input_ > 0.5] = 0
        return grad_input


class SigmoidStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return (input_ > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        res = torch.sigmoid(input_)
        return res * (1 - res) * grad_input


class TanhStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return (input_ > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input * (
            1.0 - (torch.nn.functional.tanh(input_) * torch.nn.functional.tanh(input_))
        )


smooth_step = SmoothStep.apply
smooth_sigmoid = SigmoidStep.apply
fast_sigmoid = FastSigmoid.apply
tanh_step = TanhStep.apply
