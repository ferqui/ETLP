import matplotlib.pyplot as plt
import numpy as np
import tonic
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ETLP.models import NetworkBuilder
from ETLP.utils import prediction_mostcommon

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## Dataloading
sensor_size = tonic.datasets.SHD.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=10000)

dataset_train = tonic.datasets.SHD(
    save_to="./.data", train=True, transform=frame_transform
)

dataset_test = tonic.datasets.SHD(
    save_to="./.data", train=False, transform=frame_transform
)

dl_train = DataLoader(
    dataset_train,
    shuffle=True,
    batch_size=128,
    collate_fn=tonic.collation.PadTensors(batch_first=True),
)

dl_test = DataLoader(
    dataset_test,
    shuffle=False,
    batch_size=128,
    collate_fn=tonic.collation.PadTensors(batch_first=True),
)

## Network builder
method = "BP"
network = NetworkBuilder(
    np.prod(sensor_size), 20, [256], alphas=[0.4, 0.9], method=method
).to(device)
loss_fn = torch.nn.NLLLoss()
sigmoid = torch.nn.Sigmoid()
logSoftmax = torch.nn.LogSoftmax(dim=1)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

for epoch in range(10):
    network.train()
    train_label = np.array([])
    train_prediction = np.array([])
    pbar = tqdm(dl_train)
    for frame, target in pbar:
        target = target.to(device)
        frame = torch.flatten(frame.to(device), start_dim=2)

        output_acum = []
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        network.init_state(frame)
        for t in range(frame.shape[1]):
            output = network(
                frame[:, t], torch.nn.functional.one_hot(target, num_classes=20), t
            )
            output_acum.append(output[-1])
            # loss_ = loss_fn(torch.log(sigmoid(output[-1])), target)
            loss_ = loss_fn(logSoftmax(output[-1]), target)
            ETLPgrad = torch.autograd.grad(loss_, output[0], retain_graph=True)[0]
            loss += loss_
            if method == "ETLP":
                network.grad(loss_)
        output_acum = torch.stack(output_acum, dim=1)

        loss.backward()
        if method == "ETLP":
            network.applyGrad()
        optimizer.step()

        with torch.no_grad():
            prediction = prediction_mostcommon(output_acum.cpu().numpy())
            train_label = np.append(train_label, target.cpu().numpy())
            train_prediction = np.append(train_prediction, prediction)

            accuracy = np.mean((train_prediction == train_label))

        pbar.set_postfix({"Loss": loss.cpu().item(), "Accuracy": f"{accuracy:%}"})

    network.eval()
    with torch.no_grad():
        test_label = np.array([])
        test_prediction = np.array([])
        pbar = tqdm(dl_test)
        for frame, target in pbar:
            target = target.to(device)
            frame = torch.flatten(frame.to(device), start_dim=2)

            output_acum = []
            loss = torch.tensor(0.0, device=device)
            network.init_state(frame)
            for t in range(frame.shape[1]):
                output = network(
                    frame[:, t], torch.nn.functional.one_hot(target, num_classes=20)
                )
                output_acum.append(output[-1])
                # loss_ = loss_fn(torch.log(sigmoid(output[-1])), target)
                loss_ = loss_fn(logSoftmax(output[-1]), target)
                loss += loss_
            output_acum = torch.stack(output_acum, dim=1)

            prediction = prediction_mostcommon(output_acum.cpu().numpy())
            test_label = np.append(test_label, target.cpu().numpy())
            test_prediction = np.append(test_prediction, prediction)

            accuracy = np.mean((test_prediction == test_label))

            pbar.set_postfix({"Loss": loss.item(), "Accuracy": f"{accuracy:%}"})
        tqdm.write(f"Accuracy: {accuracy:%}")
