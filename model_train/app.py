from model_train.model.model import ExampleModel
from model_train.utils.train import train
from model_train.utils.data_load import data_load

import torch
import argparse


def main(args):
    mnist_train = data_load(args.path)

    # dataset load
    train_loader = torch.utils.data.DataLoader(
        dataset=mnist_train, batch_size=args.batch_size, shuffle=True
    )

    model = ExampleModel().to(args.device)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    total_batch = len(train_loader)
    metrics = dict()
    metrics["train_avg_cost"] = 0
    metrics["train_avg_accuracy"] = 0
    metrics["valid_avg_cost"] = 0
    metrics["valid_avg_accuracy"] = 0

    train(
        epochs=args.epochs,
        total_batch=total_batch,
        metrics=metrics,
        optimizer=optimizer,
        criterion=criterion,
        model=model,
        device=args.device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        dest="path",
        type=str,
        help="/path/to/data",
        default="/tmp",
        action="store",
    )

    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        default=3,
        action="store",
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=10,
        action="store",
    )

    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=0.01,
        action="store",
    )

    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cpu",
        action="store",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        type=str,
        default="/tmp",
        action="store",
    )

    args = parser.parse_args()
    main(args)
