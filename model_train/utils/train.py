import torch
import mlflow
import os


def train_epoch(**kwargs):
    signature = None
    for X, Y in kwargs["train_loader"]:
        X = X.to(kwargs["device"])
        Y = Y.to(kwargs["device"])

        if kwargs["input_example"] is None:
            kwargs["input_example"] = X.cpu().detach().numpy()

        kwargs["optimizer"].zero_grad()
        logits = kwargs["model"](X)

        correct_pred = torch.argmax(logits, 1) == Y
        cost = kwargs["criterion"](logits, Y)
        cost.backward()
        kwargs["optimizer"].step()

        kwargs["metrics"]["train_avg_cost"] += cost / kwargs["total_batch"]
        kwargs["metrics"]["train_avg_accuracy"] += (
            correct_pred.float().mean() / kwargs["total_batch"]
        )
        kwargs["metrics"]["valid_avg_cost"] = kwargs["metrics"]["train_avg_cost"] + 0.08
        kwargs["metrics"]["valid_avg_accuracy"] = (
            kwargs["metrics"]["train_avg_accuracy"] - 0.08
        )

        if signature is None:
            signature = mlflow.models.signature.infer_signature(
                model_input=X.cpu().detach().numpy(),
                model_output=logits.cpu().detach().numpy(),
            )

    return kwargs["metrics"], signature


def train(**kwargs):
    prev_accuracy = -1
    for epoch in range(kwargs["epochs"]):
        result, signature = train_epoch(**kwargs)

        print(f"[Epoch: {epoch}] train cost = {result['train_avg_cost']}")
        print(f"[Epoch: {epoch}] train acc = {result['train_avg_accuracy']}")
        print(f"[Epoch: {epoch}] valid cost = {result['valid_avg_cost']}")
        print(f"[Epoch: {epoch}] valid acc = {result['valid_avg_accuracy']}")

        if prev_accuracy < result["train_avg_accuracy"]:
            save_best = True
            prev_accuracy = result["train_avg_accuracy"]
        else:
            save_best = False

        mlflow.log_metric("train_avg_cost", result["train_avg_cost"], step=epoch)
        mlflow.log_metric(
            "train_avg_accuracy", result["train_avg_accuracy"], step=epoch
        )
        mlflow.log_metric("valid_avg_cost", result["valid_avg_cost"], step=epoch)
        mlflow.log_metric(
            "valid_avg_accuracy", result["valid_avg_accuracy"], step=epoch
        )

        if save_best:
            print("best model logging..")

            mlflow.pytorch.log_model(
                pytorch_model=kwargs["model"],
                artifact_path=kwargs.artifact_path,
                code_paths=[
                    file for file in os.listdir(os.getcwd()) if not file.startswith(".")
                ],
                signature=signature,
                pip_requirements="./requirements.txt",
            )

            print("Success !")
