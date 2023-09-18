import math
from collections import namedtuple

import numpy as np
import torch

from .surrogate import tanh_step


class HookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, labels, y, fixed_fb_weights, train_mode):
        if train_mode in ["DFA", "sDFA", "DRTP", "ETLP"]:
            ctx.save_for_backward(input, labels, y, fixed_fb_weights)
        ctx.in1 = train_mode
        return input

    @staticmethod
    def backward(ctx, grad_output):
        train_mode = ctx.in1
        if train_mode == "BP":
            return grad_output, None, None, None, None
        elif train_mode == "shallow":
            grad_output.data.zero_()
            return grad_output, None, None, None, None

        input, labels, y, fixed_fb_weights = ctx.saved_variables
        if train_mode == "DFA":
            grad_output_est = (
                (y - labels)
                .mm(fixed_fb_weights.view(-1, np.prod(fixed_fb_weights.shape[1:])))
                .view(grad_output.shape)
            )
        elif train_mode == "sDFA":
            grad_output_est = (
                torch.sign(y - labels)
                .mm(fixed_fb_weights.view(-1, np.prod(fixed_fb_weights.shape[1:])))
                .view(grad_output.shape)
            )
        elif train_mode == "DRTP" or train_mode == "ETLP":
            grad_output_est = (
                labels.float()
                .mm(fixed_fb_weights.view(-1, np.prod(fixed_fb_weights.shape[1:])))
                .view(grad_output.shape)
            )
        else:
            raise NameError(
                "=== ERROR: training mode " + str(train_mode) + " not supported"
            )

        return grad_output_est, None, None, None, None


trainingHook = HookFunction.apply


class TrainingHook(torch.nn.Module):
    def __init__(self, dim_hook, train_mode):
        super(TrainingHook, self).__init__()
        self.train_mode = train_mode
        assert train_mode in ["BP", "DFA", "DRTP", "ETLP", "sDFA", "shallow"], (
            "=== ERROR: Unsupported hook training mode " + train_mode + "."
        )

        if self.train_mode in ["DFA", "DRTP", "sDFA", "ETLP"]:
            self.fixed_fb_weights = torch.nn.Parameter(
                torch.Tensor(torch.Size(dim_hook))
            )
            self.reset_weights()
        else:
            self.fixed_fb_weights = None

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False

    def forward(self, input, labels, y):
        return trainingHook(input, labels, y, self.fixed_fb_weights, self.train_mode)

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.train_mode + ")"


class LIFLayer(torch.nn.Module):
    NeuronState = namedtuple("NeuronState", ["V", "S", "pre_trace", "post_trace"])

    def __init__(self, in_features, out_features, alpha=0.9, activation=tanh_step):
        super(LIFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Win = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features), requires_grad=False)
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.state = None
        self.activation = activation
        self.gradient = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.Win, a=math.sqrt(5))
        torch.nn.init.constant_(self.bias, 1.0)

    def init_state(self, input_):
        self.state = self.NeuronState(
            V=torch.zeros(input_.shape[0], self.out_features, device=input_.device),
            S=torch.zeros(input_.shape[0], self.out_features, device=input_.device),
            pre_trace=torch.zeros(
                input_.shape[0], input_.shape[-1], device=input_.device
            ),
            post_trace=torch.zeros(
                input_.shape[0], self.out_features, device=input_.device
            ),
        )

    def detach(self):
        for state in self.state:
            state.detach_()

    @torch.no_grad()
    def grad(self, L):
        gradient = torch.einsum(
            "bo,bi->oi", L * self.state.post_trace, self.state.pre_trace
        )
        try:
            self.gradient += gradient
        except:
            self.gradient = gradient

    @torch.no_grad()
    def applyGrad(self):
        self.Win.grad = self.gradient

    def forward(self, input_):
        if self.state is None:
            self.init_state(input_)
        state = self.state

        with torch.no_grad():
            pre_trace = state.pre_trace * self.alpha + input_
            # pre_trace = state.pre_trace * self.alpha * ((1 - S) - S * post_trace) + input_

        V = state.V * self.alpha * (1 - state.S)
        V = V + torch.nn.functional.linear(input_, self.Win)
        S = self.activation(V - self.bias)

        with torch.no_grad():
            post_trace = 1.0 - (
                torch.nn.functional.tanh(V - self.bias)
                * torch.nn.functional.tanh(V - self.bias)
            )

        self.state = self.NeuronState(
            V=V, S=S, pre_trace=pre_trace, post_trace=post_trace
        )

        return S


class LeakyLayer(torch.nn.Module):
    NeuronState = namedtuple("NeuronState", ["V"])

    def __init__(self, in_features, out_features, alpha=0.9):
        super(LeakyLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Win = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.state = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.Win, a=math.sqrt(5))

    def init_state(self, input_):
        self.state = self.NeuronState(
            V=torch.zeros(input_.shape[0], self.out_features, device=input_.device),
        )

    def detach(self):
        for state in self.state:
            state.detach_()

    def forward(self, input_):
        if self.state is None:
            self.init_state(input_)

        state = self.state
        V = state.V * self.alpha
        V = V + torch.nn.functional.linear(input_, self.Win)

        self.state = self.NeuronState(V=V)

        return V


class NetworkBuilder(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        Nhid=[1],
        alphas=[0.9, 0.9],
        outputclass=LeakyLayer,
        method="BP",
    ):
        super(NetworkBuilder, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        Nhid = [input_shape] + Nhid
        self.hooks = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(Nhid)):
            self.layers.append(LIFLayer(Nhid[i - 1], Nhid[i]))
            self.hooks.append(TrainingHook([output_shape, Nhid[i]], train_mode=method))

        self.output_layer = outputclass(Nhid[-1], output_shape)
        self.method = method
        self.y = None

    def detach(self):
        for layer in self.layers:
            layer.detach()
        self.output_layer.detach()

    def init_state(self, input_):
        for layer in self.layers:
            layer.init_state(input_)
        self.output_layer.init_state(input_)

    @torch.no_grad()
    def grad(self, L):
        for layer in self.layers:
            layerGrad = torch.autograd.grad(L, layer.state.S, retain_graph=True)[0]
            layer.grad(layerGrad)

    @torch.no_grad()
    def applyGrad(self):
        for layer in self.layers:
            layer.applyGrad()

    def forward(self, input_, target=None, t=0):
        if (self.method == "DFA") or (self.method == "sDFA"):
            y = torch.zeros(input_.shape[0], self.output_shape, device=input_.device)
            y.requires_grad = False
        else:
            y = None

        out = [[] for _ in range(len(self.layers) + 1)]
        for layer_id, (layer, hook) in enumerate(zip(self.layers, self.hooks)):
            layerOutput_ = layer(input_)
            input_ = hook(layerOutput_, target, y)
            out[layer_id] = layerOutput_
        output = self.output_layer(input_)
        out[-1] = output

        if input_.requires_grad and (y is not None):
            y.data.copy_(output.data)  # in-place update, only happens with (s)DFA
        return out
