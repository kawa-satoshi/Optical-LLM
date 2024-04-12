import time

import mlflow
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
    device: torch.device,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
):
    loss = 0.0
    num_correct = 0
    num_data = 0
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        y = model(images)

        optimizer.zero_grad()
        loss_t = loss_fn(y, labels)
        loss_t.backward()
        optimizer.step()

        loss += loss_t.item()
        num_correct += (y.max(1)[1] == labels).sum().item()
        num_data += images.size()[0]
    return loss, num_correct / num_data


def test(
    device: torch.device, model: torch.nn.Module, loss_fn: torch.nn.Module, data_loader: torch.utils.data.DataLoader
):
    loss = 0.0
    num_correct = 0
    num_data = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            y = model(images)
            loss_t = loss_fn(y, labels).item()
            loss += loss_t
            num_correct += (y.max(1)[1] == labels).sum().item()
            num_data += images.size()[0]
    return loss, num_correct / num_data


def log(
    writer: SummaryWriter, name: str, train_or_test: str, analog_or_digital: str, epoch: int, loss: float, acc: float
):
    if writer is None:
        print(f"{analog_or_digital} {train_or_test} loss = {loss:.3f}, {train_or_test} acc = {acc:.3f}")
    else:
        writer.add_scalars(f"loss/{train_or_test}", {name + "/" + analog_or_digital: loss}, epoch)
        writer.add_scalars(f"accuracy/{train_or_test}", {name + "/" + analog_or_digital: acc}, epoch)


def set_analog_mode(model, value: bool):
    model.analog_mode = value
    for layer in model.modules():
        if layer is model:
            continue

        try:
            layer.set_analog_mode(value)
        except AttributeError:
            continue


def log_metric(key, value, step, no_mlflow=False):
    if no_mlflow:
        return

    mlflow.log_metric(key, value, step=step)


def run(
    name: str,
    device: torch.DeviceObjType,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    test_loader,
    train_mode: str = "analog",
    epochs: int = 20,
    writer: SummaryWriter = None,
    save_interval: int = None,
    filename_base: str = "results/cifar10/vit_digital",
    no_mlflow: bool = False,
):
    if save_interval is not None and filename_base is not None:
        print("Model will be saved.")

    start = time.time()
    model.eval()

    if epochs != 0:
        set_analog_mode(model, False)
        train_loss, train_acc = test(device, model, loss_fn, train_loader)
        log(writer, name, "train", "digital", 0, train_loss, train_acc)
        log_metric("train_loss", train_loss, step=0, no_mlflow=no_mlflow)
        log_metric("train_acc", train_acc, step=0, no_mlflow=no_mlflow)

        test_loss, test_acc = test(device, model, loss_fn, test_loader)
        log(writer, name, "test", "digital", 0, test_loss, test_acc)
        log_metric("digital_test_loss", test_loss, step=0, no_mlflow=no_mlflow)
        log_metric("digital_test_acc", test_acc, step=0, no_mlflow=no_mlflow)

    set_analog_mode(model, True)
    test_loss, test_acc = test(device, model, loss_fn, test_loader)
    log(writer, name, "test", "analog", 0, test_loss, test_acc)
    log_metric("analog_test_loss", test_loss, step=0, no_mlflow=no_mlflow)
    log_metric("analog_test_acc", test_acc, step=0, no_mlflow=no_mlflow)

    for epoch in range(1, epochs + 1):
        model.train()
        set_analog_mode(model, train_mode == "analog")
        train_loss, train_acc = train(device, model, loss_fn, optimizer, train_loader)
        log(writer, name, "train", "digital", epoch, train_loss, train_acc)
        log_metric("train_loss", train_loss, step=epoch, no_mlflow=no_mlflow)
        log_metric("train_acc", train_acc, step=epoch, no_mlflow=no_mlflow)

        model.eval()
        set_analog_mode(model, False)
        test_loss, test_acc = test(device, model, loss_fn, test_loader)
        log(writer, name, "test", "digital", epoch, test_loss, test_acc)
        log_metric("digital_test_loss", test_loss, step=epoch, no_mlflow=no_mlflow)
        log_metric("digital_test_acc", test_acc, step=epoch, no_mlflow=no_mlflow)

        set_analog_mode(model, True)
        test_loss, test_acc = test(device, model, loss_fn, test_loader)
        log(writer, name, "test", "analog", epoch, test_loss, test_acc)
        log_metric("analog_test_loss", test_loss, step=epoch, no_mlflow=no_mlflow)
        log_metric("analog_test_acc", test_acc, step=epoch, no_mlflow=no_mlflow)

        print("Epoch {} done. Elapsed: {:.3f} s".format(epoch, time.time() - start))

        if save_interval is not None and filename_base is not None and epoch % save_interval == 0:
            torch.save(model.state_dict(), "{}-e{}.model".format(filename_base, epoch))
            torch.save(optimizer.state_dict(), "{}-e{}.optim".format(filename_base, epoch))
            print("Saved the model.")
