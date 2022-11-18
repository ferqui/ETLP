# -*- coding: utf-8 -*-
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset, SlicedDataset
from tonic.slicers import SliceByTime

from utils import *
from models import *

import numpy as np

import argparse

device = torch.device("cuda")

"""# Train
Parameter definitions
"""
parser = argparse.ArgumentParser(
    description="Event-based Three factor Learning Plasticity (ETLP)"
)

# Learning parameters
parser.add_argument("--seed", type=int, default=0, help="Learning rate")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument(
    "--method",
    type=str,
    default="ETLP",
    choices=["BPTT", "EPROP", "ETLP"],
    help="Choose between different learning rules",
)
parser.add_argument(
    "--voltage_reg", type=float, default=0.0, help="Voltage regularization"
)
parser.add_argument(
    "--weight_L1", type=float, default=0.0, help="Weight L1 regularization"
)
parser.add_argument(
    "--weight_L2", type=float, default=0.0, help="Weight L2 regularization"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="SHD",
    choices=["N-MNIST", "SHD", "gesture"],
    help="Choose between different datasets",
)

# Network parameters
parser.add_argument(
    "--n_rec", type=int, default=200, help="Number of recurrent neurons"
)
parser.add_argument(
    "--tau_v",
    type=float,
    default=80.0,
    help="Membrane decay constant (ms) for recurrent neurons",
)
parser.add_argument(
    "--tau_a", type=float, default=10.0, help="Threshold adaptation time constant (ms)"
)
parser.add_argument(
    "--tau_o",
    type=float,
    default=80.0,
    help="Membrane decay constant (ms) for output neurons",
)
parser.add_argument(
    "--theta", type=float, default=5.0, help="Adaptative threshold increase"
)
parser.add_argument("--thr", type=float, default=1.0, help="Neurons threshold")
parser.add_argument(
    "--n_ref", type=int, default=5.0, help="Neurons refractory period (ms)"
)
parser.add_argument("--dt", type=float, default=1, help="Simulation timestep (ms)")
parser.add_argument("--recurrent", action="store_true", help="Use explicit recurrency")
parser.add_argument("--train_rec", action="store_true", help="Train recurrent layer")

args = parser.parse_args()

DATASET = args.dataset

np.random.seed(args.seed)
torch.manual_seed(args.seed)

cache_dir = os.path.expanduser("./data")

if DATASET == "SHD":

    args.n_in = 700
    args.n_out = 20

    sensor_size = tonic.datasets.SHD.sensor_size
    transform = transforms.ToFrame(sensor_size=sensor_size, time_window=args.dt * 1000)

    trainset = tonic.datasets.SHD(save_to=cache_dir, train=True, transform=transform)
    testset = tonic.datasets.SHD(save_to=cache_dir, train=False, transform=transform)

    trainset = DiskCachedDataset(
        trainset, os.path.join(cache_dir, "cached/SHD/train"), reset_cache=False
    )
    testset = DiskCachedDataset(
        testset, os.path.join(cache_dir, "cached/SHD/test"), reset_cache=False
    )

    # idx = np.argsort(np.random.rand(len(trainset)))
    # trainset = torch.utils.data.Subset(trainset, idx[:640])

    # idx = np.argsort(np.random.rand(len(testset)))
    # testset = torch.utils.data.Subset(testset, idx[:640])

    train_dl = DataLoader(
        trainset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )
    test_dl = DataLoader(
        testset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )

elif DATASET == "N-MNIST":
    args.n_in = 2 * 34 * 34
    args.n_out = 10

    sensor_size = tonic.datasets.NMNIST.sensor_size
    transform = transforms.ToFrame(sensor_size=sensor_size, time_window=args.dt * 1000)

    trainset = tonic.datasets.NMNIST(
        save_to=cache_dir, train=True, transform=transform, first_saccade_only=True
    )
    testset = tonic.datasets.NMNIST(
        save_to=cache_dir, train=False, transform=transform, first_saccade_only=True
    )

    trainset = DiskCachedDataset(
        trainset, os.path.join(cache_dir, "cached/NMNIST/train"), reset_cache=False
    )
    testset = DiskCachedDataset(
        testset, os.path.join(cache_dir, "cached/NMNIST/test"), reset_cache=False
    )

    # idx = np.argsort(np.random.rand(len(trainset)))
    # trainset = torch.utils.data.Subset(trainset, idx[:6400])

    # idx = np.argsort(np.random.rand(len(testset)))
    # testset = torch.utils.data.Subset(testset, idx[:6400])

    train_dl = DataLoader(
        trainset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )
    test_dl = DataLoader(
        testset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )

elif DATASET == "gesture":

    sensor_size = tonic.datasets.DVSGesture.sensor_size
    transform = transforms.ToFrame(sensor_size=sensor_size, time_window=args.dt * 1000)

    args.n_in = np.prod(sensor_size)
    args.n_out = 11

    trainset = tonic.datasets.DVSGesture(save_to=cache_dir, train=True)
    testset = tonic.datasets.DVSGesture(save_to=cache_dir, train=False)

    slicing_time_window = 100000  # microseconds
    slicer = SliceByTime(time_window=slicing_time_window)
    trainset = SlicedDataset(
        trainset,
        slicer=slicer,
        metadata_path=os.path.join(cache_dir, "metadata/DVSGesture/train"),
        transform=transform,
    )
    testset = SlicedDataset(
        testset,
        slicer=slicer,
        metadata_path=os.path.join(cache_dir, "metadata/DVSGesture/test"),
        transform=transform,
    )

    trainset = DiskCachedDataset(
        trainset, os.path.join(cache_dir, "cached/DVSGesture/train"), reset_cache=True
    )
    testset = DiskCachedDataset(
        testset,
        os.path.join(cache_dir, "data/cached/DVSGesture/test"),
        reset_cache=True,
    )

    train_dl = DataLoader(
        trainset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )
    test_dl = DataLoader(
        testset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )

SpikeFunction.scale = 0.3 / args.thr

"""Model and optimizer creation"""
model = Network(n_in=args.n_in, n_rec=args.n_rec, n_out=args.n_out, args=args).to(
    device
)

optimizer = optim.Adamax(
    model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_L2
)
loss_fn = lambda x, y: 0.5 * torch.sum((x - y) ** 2)

writer = SummaryWriter()

print("Loading data in cache...")
for _ in tqdm(train_dl):
    pass

for _ in tqdm(test_dl):
    pass

print("Training loop...")
total = None
for epoch in range(args.epochs):
    train(args, model, train_dl, optimizer, loss_fn, device, total=total)

    (
        acc_train,
        loss_train,
        input_train,
        recurrent_v_train,
        recurrent_s_train,
        output_v_train,
        output_train,
        target_train,
        pred_train,
    ) = test(args, model, train_dl, loss_fn, device, total=total)

    (
        acc_test,
        loss_test,
        input_test,
        recurrent_v_test,
        recurrent_s_test,
        output_v_test,
        output_test,
        target_test,
        pred_test,
    ) = test(args, model, test_dl, loss_fn, device, total=total)

    (
        input_train,
        recurrent_s_train,
        output_train,
        recurrent_v_train,
        output_v_train,
    ) = plot_images(
        input_train, recurrent_v_train, recurrent_s_train, output_v_train, output_train
    )
    (
        input_test,
        recurrent_s_test,
        output_test,
        recurrent_v_test,
        output_v_test,
    ) = plot_images(
        input_test, recurrent_v_test, recurrent_s_test, output_v_test, output_test
    )

    print("Epoch " + str(epoch))
    print("Training: accuracy - {0} , loss - {1}".format(acc_train, loss_train))
    print("Testing: accuracy - {0} , loss - {1}".format(acc_test, loss_test))

    with torch.no_grad():
        writer.add_scalar("Train/Loss", loss_train, epoch)
        writer.add_scalar("Test/Loss", loss_test, epoch)
        writer.add_scalar("Train/Accuracy", acc_train, epoch)
        writer.add_scalar("Test/Accuracy", acc_test, epoch)

        writer.add_figure("Train/Input", input_train, epoch)
        writer.add_figure("Train/Recurrent", recurrent_s_train, epoch)
        writer.add_figure("Train/Output", output_train, epoch)
        writer.add_figure("Train/Recurrent (voltage)", recurrent_v_train, epoch)
        writer.add_figure("Train/Output (voltage)", output_v_train, epoch)

        writer.add_figure("Test/Input", input_test, epoch)
        writer.add_figure("Test/Recurrent", recurrent_s_test, epoch)
        writer.add_figure("Test/Output", output_test, epoch)
        writer.add_figure("Test/Recurrent (voltage)", recurrent_v_test, epoch)
        writer.add_figure("Test/Output (voltage)", output_v_test, epoch)

        writer.flush()
