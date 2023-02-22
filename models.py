import torch
import torch.nn as nn

import numpy as np
from collections import namedtuple


class SpikeFunction(torch.autograd.Function):

    scale = 0.3

    @staticmethod
    def pseudo_derivative(v):
        # return 1.0 / (10 * torch.abs(v) + 1.0) ** 2
        return torch.maximum(1 - torch.abs(v), torch.tensor(0)) * SpikeFunction.scale

    @staticmethod
    def forward(ctx, v_scaled):
        ctx.save_for_backward(v_scaled)
        return (v_scaled > 0).type(v_scaled.dtype)

    @staticmethod
    def backward(ctx, dy):
        (v_scaled,) = ctx.saved_tensors

        dE_dz = dy
        dz_dv_scaled = SpikeFunction.pseudo_derivative(v_scaled)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled


activation = SpikeFunction.apply


class Network(nn.Module):
    NeuronState = namedtuple(
        "NeuronState",
        (
            "V_rec",
            "S_rec",
            "R_rec",
            "A_rec",
            "V_out",
            "S_out",
            "e_trace_in",
            "e_trace_rec",
            "epsilon_v_in",
            "epsilon_v_rec",
            "epsilon_v_out",
            "epsilon_a_in",
            "epsilon_a_rec",
        ),
    )

    def __init__(self, n_in, n_rec, n_out, args):
        super(Network, self).__init__()

        self.dt = args.dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.n_refractory = args.n_ref
        self.recurrent = args.recurrent
        self.keep_trace = args.method == "EPROP"

        # Weight matrix creation
        self.W_in = torch.nn.Parameter(
            torch.tensor(0.2 * np.random.randn(n_in, n_rec) / np.sqrt(n_in)).float(),
            requires_grad=True,
        )
        if self.recurrent:
            recurrent_weights = 0.2 * np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)
            self.W_rec = torch.nn.Parameter(
                torch.tensor(
                    recurrent_weights - recurrent_weights * np.eye(n_rec, n_rec)
                ).float(),
                requires_grad=args.train_rec,
            )
        self.W_out = torch.nn.Parameter(
            torch.tensor(np.random.randn(n_rec, n_out) / np.sqrt(n_rec)).float(),
            requires_grad=True,
        )
        self.register_buffer(
            "b_out",
            torch.tensor(np.random.randn(n_rec, n_out) / np.sqrt(n_rec)).float(),
        )

        # Self recurrency
        self.register_buffer("identity_diag_rec", torch.eye(n_rec, n_rec))

        # Parameters creation
        distribution = torch.distributions.gamma.Gamma(3, 3 / args.tau_v)
        tau_v = distribution.rsample((1, n_rec)).clamp(3, 100)
        self.register_buffer("decay_v", torch.exp(-args.dt / tau_v).float())
        # self.register_buffer('decay_v', torch.tensor(np.exp(-dt/tau_v)).float())
        self.register_buffer(
            "decay_o", torch.tensor(np.exp(-args.dt / args.tau_o)).float()
        )
        self.register_buffer(
            "decay_a", torch.tensor(np.exp(-args.dt / args.tau_a)).float()
        )
        self.register_buffer("thr", torch.tensor(args.thr).float())
        self.register_buffer("theta", torch.tensor(args.theta).float())

        self.state = None

    def initialize_state(self, input):
        state = self.NeuronState(
            V_rec=torch.zeros(input.shape[0], self.n_rec, device=input.device),
            S_rec=torch.zeros(input.shape[0], self.n_rec, device=input.device),
            R_rec=torch.zeros(input.shape[0], self.n_rec, device=input.device),
            A_rec=torch.zeros(input.shape[0], self.n_rec, device=input.device),
            V_out=torch.zeros(input.shape[0], self.n_out, device=input.device),
            S_out=torch.zeros(input.shape[0], self.n_out, device=input.device),
            e_trace_in=torch.zeros(
                input.shape[0], self.n_in, self.n_rec, device=input.device
            ),
            e_trace_rec=torch.zeros(
                input.shape[0], self.n_rec, self.n_rec, device=input.device
            ),
            epsilon_v_in=torch.zeros(
                input.shape[0], self.n_in, self.n_rec, device=input.device
            ),
            epsilon_v_rec=torch.zeros(
                input.shape[0], self.n_rec, self.n_rec, device=input.device
            ),
            # epsilon_v_in  = torch.zeros(input.shape[0], self.n_in, device=input.device),
            # epsilon_v_rec = torch.zeros(input.shape[0], self.n_rec, device=input.device),
            epsilon_v_out=torch.zeros(input.shape[0], self.n_rec, device=input.device),
            epsilon_a_in=torch.zeros(
                input.shape[0], self.n_in, self.n_rec, device=input.device
            ),
            epsilon_a_rec=torch.zeros(
                input.shape[0], self.n_rec, self.n_rec, device=input.device
            ),
        )
        return state

    def reset(self):
        self.state = None

    def forward(self, input):
        if self.state is None:
            self.state = self.initialize_state(input)

        # Neuron parameters
        V_rec = self.state.V_rec
        S_rec = self.state.S_rec

        V_out = self.state.V_out
        S_out = self.state.S_out

        R_rec = self.state.R_rec  # refractory period
        A_rec = self.state.A_rec  # Threshold adaptation

        # ETLP parameters
        e_trace_in = self.state.e_trace_in
        epsilon_a_in = self.state.epsilon_a_in
        epsilon_v_in = self.state.epsilon_v_in
        e_trace_rec = self.state.e_trace_rec
        epsilon_v_rec = self.state.epsilon_v_rec
        epsilon_a_rec = self.state.epsilon_a_rec
        epsilon_v_out = self.state.epsilon_v_out

        with torch.no_grad():
            A = self.thr + self.theta * A_rec
            psi = SpikeFunction.pseudo_derivative((V_rec - A) / self.thr)
            # epsilon_a_in = psi[:,None,:] * epsilon_v_in[:,:,None] + (self.decay_a  - psi[:,None,:]*self.theta)*epsilon_a_in
            epsilon_a_in = (
                psi[:, None, :] * epsilon_v_in
                + (self.decay_a - psi[:, None, :] * self.theta) * epsilon_a_in
            )
            if self.recurrent:
                # epsilon_a_rec = psi[:,None,:] * epsilon_v_rec[:,:,None] + (self.decay_a  - psi[:,None,:]*self.theta)*epsilon_a_rec
                epsilon_a_rec = (
                    psi[:, None, :] * epsilon_v_rec
                    + (self.decay_a - psi[:, None, :] * self.theta) * epsilon_a_rec
                )

        # Threshold adaptation
        A_rec = self.decay_a * A_rec + S_rec
        A = self.thr + A_rec * self.theta

        # Detach previous spike for recurrency and reset
        S_rec = S_rec.detach()

        # Current calculation
        if self.recurrent:
            I_in = torch.mm(input, self.W_in) + torch.mm(S_rec, self.W_rec)
        else:
            I_in = torch.mm(input, self.W_in)
        # I_reset = S_rec * self.thr

        # Recurrent neurons update
        # V_rec_new = (self.decay_v * V_rec + I_in) * (1-S_rec)
        V_rec_new = self.decay_v * V_rec + I_in - self.thr * S_rec

        # Spike generation
        is_refractory = R_rec > 0
        zeros_like_spikes = torch.zeros_like(S_rec)
        S_rec_new = torch.where(
            is_refractory, zeros_like_spikes, activation((V_rec_new - A) / self.thr)
        )
        R_rec_new = R_rec + self.n_refractory * S_rec_new - 1
        R_rec_new = torch.clip(R_rec_new, 0.0, self.n_refractory).detach()

        # Forward pass of the data to output weights
        I_out = torch.mm(S_rec_new, self.W_out)

        # Recurrent neurons update
        V_out_new = self.decay_o * V_out + I_out - self.thr * S_out
        # V_out_new = (self.decay_o * V_out + I_out) * (1-S_out)
        S_out_new = activation((V_out - self.thr) / self.thr)

        with torch.no_grad():
            if input.is_sparse:
                epsilon_v_in = (
                    self.decay_v[:, None, :] * epsilon_v_in
                    + input.to_dense()[:, :, None]
                )
                # epsilon_v_in  = self.decay_v * epsilon_v_in + input.to_dense()
            else:
                epsilon_v_in = (
                    self.decay_v[:, None, :] * epsilon_v_in + input[:, :, None]
                )
                # epsilon_v_in  = self.decay_v * epsilon_v_in + input
            if self.recurrent:
                epsilon_v_rec = (
                    self.decay_v[:, None, :] * epsilon_v_rec + S_rec[:, :, None]
                )
                # epsilon_v_rec = self.decay_v * epsilon_v_rec + S_rec
            epsilon_v_out = self.decay_o * epsilon_v_out + S_rec_new

            v_scaled = (V_rec_new - A) / self.thr
            is_refractory = R_rec > 0
            psi_no_ref = SpikeFunction.pseudo_derivative(v_scaled)
            psi = torch.where(is_refractory, torch.zeros_like(psi_no_ref), psi_no_ref)

            if self.keep_trace:
                e_trace_in = e_trace_in * self.decay_o + (
                    psi[:, None, :] * (epsilon_v_in - self.theta * epsilon_a_in)
                )
                if self.recurrent:
                    e_trace_rec = e_trace_rec * self.decay_o + (
                        psi[:, None, :] * (epsilon_v_rec - self.theta * epsilon_a_rec)
                    )
            else:
                e_trace_in = psi[:, None, :] * (
                    epsilon_v_in - self.theta * epsilon_a_in
                )
                if self.recurrent:
                    e_trace_rec = psi[:, None, :] * (
                        epsilon_v_rec - self.theta * epsilon_a_rec
                    )

            # e_trace_in = e_trace_in * self.decay_o + (psi[:,None,:] * (epsilon_v_in[:,:,None] - self.theta*epsilon_a_in)) # psi[:,None,:] * epsilon_v_in
            # e_trace_rec = e_trace_rec * self.decay_o + (psi[:,None,:] * (epsilon_v_rec[:,:,None] - self.theta*epsilon_a_rec)) # psi[:,None,:] * epsilon_v_rec
            # e_trace_rec -= self.identity_diag_rec[None,:,:] * e_trace_rec # No self recurrency

        new_state = self.NeuronState(
            V_rec=V_rec_new,
            S_rec=S_rec_new,
            R_rec=R_rec_new,
            A_rec=A_rec,
            V_out=V_out_new,
            S_out=S_out_new,
            e_trace_in=e_trace_in.detach(),
            e_trace_rec=e_trace_rec.detach(),
            epsilon_v_in=epsilon_v_in.detach(),
            epsilon_v_rec=epsilon_v_rec.detach(),
            epsilon_v_out=epsilon_v_out.detach(),
            epsilon_a_in=epsilon_a_in.detach(),
            epsilon_a_rec=epsilon_a_rec.detach(),
        )

        self.state = new_state

        return S_out_new

    def detach(self):
        for state in self.state:
            state.detach_()
