import torch
import torch.nn.functional as F

import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from models import SpikeFunction

from tqdm import tqdm


def train(args, model, dataloader, optimizer, loss_fn, device, total=None):
    model.train()
    for x_local, y_local in tqdm(dataloader, total=total, postfix="Training"):
        input = x_local.to(device)
        target = y_local.to(device)

        if len(input.shape) > 3:
            input = input.view(input.shape[0], input.shape[1], -1)
        if len(target.shape) > 1:
            target = target[:, 0].max(1)[1]

        if args.method == "EPROP":
            gradient_in = torch.zeros_like(model.W_in)
            if args.train_rec:
                gradient_rec = torch.zeros_like(model.W_rec)
            gradient_out = torch.zeros_like(model.W_out)

        local_loss = 0
        model.reset()
        for t in range(input.size(0)):
            out = model(input[t])
            local_loss += loss_fn(
                out, F.one_hot(target, num_classes=args.n_out).float()
            )
            local_loss += (
                args.voltage_reg
                * 0.5
                * torch.sum(
                    model.state.V_rec**2 * (torch.abs(model.state.V_rec) > model.thr)
                )
            )

            if args.method == "ETLP":
                if np.random.rand() < model.dt * 100 * 1e-3:
                    optimizer.zero_grad()
                    with torch.no_grad():
                        labels = F.one_hot(target, num_classes=args.n_out).float()
                        error = out - labels
                        learning_signals = torch.mm(
                            -labels, model.b_out.T
                        ) + args.voltage_reg * model.state.V_rec * torch.logical_or(
                            model.state.V_rec > model.thr,
                            model.state.V_rec < -model.thr,
                        )

                        v_scaled = (model.state.V_out - model.thr) / model.thr
                        psi = SpikeFunction.pseudo_derivative(v_scaled)
                        e_trace = (
                            psi[:, None, :] * model.state.epsilon_v_out[:, :, None]
                        )

                        model.W_in.grad = torch.einsum(
                            "bj,bij->ij", learning_signals, model.state.e_trace_in
                        ) + args.weight_L1 * torch.sign(model.W_in)
                        if args.train_rec:
                            # W_rec_grad = torch.einsum('bj,bij->ij', learning_signals, model.state.e_trace_rec) + args.weight_L1 * torch.sign(model.W_rec)
                            # model.W_rec.grad = W_rec_grad - W_rec_grad*model.identity_diag_rec # No self recurrency
                            model.W_rec.grad = torch.einsum(
                                "bj,bij->ij", learning_signals, model.state.e_trace_rec
                            ) + args.weight_L1 * torch.sign(model.W_rec)
                        model.W_out.grad = torch.einsum("bj,bij->ij", error, e_trace)
                    optimizer.step()
                    # with torch.no_grad():
                    #    model.W_in.data = torch.clamp(model.W_in.data, -0.1, 0.1)
                    model.detach()
            elif args.method == "EPROP":
                with torch.no_grad():
                    labels = F.one_hot(target, num_classes=args.n_out).float()
                    error = out - labels
                    learning_signals = torch.mm(
                        error, model.b_out.T
                    ) + args.voltage_reg * model.state.V_rec * (
                        torch.abs(model.state.V_rec) > model.thr
                    )

                    v_scaled = (model.state.V_out - model.thr) / model.thr
                    psi = SpikeFunction.pseudo_derivative(v_scaled)
                    e_trace = psi[:, None, :] * model.state.epsilon_v_out[:, :, None]

                    gradient_in += torch.einsum(
                        "bj,bij->ij", learning_signals, model.state.e_trace_in
                    )
                    if args.train_rec:
                        gradient_rec += torch.einsum(
                            "bj,bij->ij", learning_signals, model.state.e_trace_rec
                        )
                    gradient_out += torch.einsum("bj,bij->ij", error, e_trace)
                model.detach()

        loss_val = local_loss + args.weight_L1 * torch.sum(torch.abs(model.W_in))

        if args.method == "EPROP":
            optimizer.zero_grad()
            with torch.no_grad():
                model.W_in.grad = gradient_in + args.weight_L1 * torch.sign(model.W_in)
                if args.train_rec:
                    model.W_rec.grad = gradient_rec + args.weight_L1 * torch.sign(
                        model.W_rec
                    )
                # with torch.no_grad():
                #    # No self recursion
                #    model.W_rec.grad = model.W_rec.grad - model.W_rec.grad*model.identity_diag_rec
                model.W_out.grad = gradient_out
            optimizer.step()
        elif args.method == "BPTT":
            optimizer.zero_grad()
            loss_val.backward()
            # with torch.no_grad():
            #    # No self recursion
            #    model.W_rec.grad = model.W_rec.grad - model.W_rec.grad*model.identity_diag_rec
            optimizer.step()


def test(args, model, dataloader, loss_fn, device, total=None):
    total_acc = []
    total_loss = []
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for x_local, y_local in tqdm(dataloader, total=total, postfix="Testing"):
            input = x_local.to(device)
            target = y_local.to(device)

            if len(input.shape) > 3:
                input = input.view(input.shape[0], input.shape[1], -1)
            if len(target.shape) > 1:
                target = target[:, 0].max(1)[1]

            output = []
            output_v = []
            recurrent_v = []
            recurrent_s = []
            local_loss = 0
            model.reset()
            for t in range(input.size(0)):
                out = model(input[t])
                output.append(out)
                output_v.append(model.state.V_out)
                recurrent_v.append(model.state.V_rec)
                recurrent_s.append(model.state.S_rec)
                local_loss += loss_fn(
                    out, F.one_hot(target, num_classes=args.n_out).float()
                )

            output = torch.stack(output, dim=0)
            output_v = torch.stack(output_v, dim=0)
            recurrent_v = torch.stack(recurrent_v, dim=0)
            recurrent_s = torch.stack(recurrent_s, dim=0)

            loss_val = local_loss  # + torch.sum(torch.abs(model.W_in)) + torch.sum(torch.abs(model.W_rec))

            num_spikes = torch.sum(output, 0)
            _, pred = torch.max(num_spikes, 1)

            acc = (pred == target).float().mean()

            y_pred.extend(pred.cpu().numpy())  # Save Prediction
            y_true.extend(target.cpu().numpy())  # Save Truth

            total_acc.append(acc)
            total_loss.append(loss_val)

        total_acc = torch.stack(total_acc)
        total_loss = torch.stack(total_loss)

    return (
        total_acc.mean(),
        total_loss.mean(),
        input.to_dense() if input.is_sparse else input,
        recurrent_v,
        recurrent_s,
        output_v,
        output,
        y_true,
        y_pred,
    )


def plot_images(input_s, hid_v, hid_s, out_v, out_s):
    idx = np.random.randint(0, input_s.shape[1])

    spk = input_s[:, idx].cpu().numpy()
    event_times, event_ids = np.where(spk)
    # fig_input = px.scatter(x=event_times, y=event_ids, labels={"x": "t", "y": "idx"})
    fig_input = plt.figure()
    plt.scatter(x=event_times, y=event_ids, s=0.5)

    spk = hid_s[:, idx].cpu().numpy()
    event_times, event_ids = np.where(spk)
    try:
        # fig_hid = px.scatter(x=event_times, y=event_ids, labels={"x": "t", "y": "idx"})
        fig_hid = plt.figure()
        plt.scatter(x=event_times, y=event_ids, s=0.5)
    except:
        fig_hid = plt.figure()
        plt.plot()

    spk = out_s[:, idx].cpu().numpy()
    event_times, event_ids = np.where(spk)
    try:
        # fig_out = px.scatter(x=event_times, y=event_ids, labels={"x": "t", "y": "idx"})
        fig_out = plt.figure()
        plt.scatter(x=event_times, y=event_ids, s=0.5)
    except:
        fig_out = plt.figure()
        plt.plot()

    fig_v_hid = plt.figure()
    plt.plot(hid_v[:, idx].cpu().numpy())

    fig_v_out = plt.figure()
    plt.plot(out_v[:, idx].cpu().numpy())

    return fig_input, fig_hid, fig_out, fig_v_hid, fig_v_out
