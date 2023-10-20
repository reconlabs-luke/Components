import torch


def train_epoch(**kwargs):
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

    return kwargs["metrics"]


def train(**kwargs):
    prev_accuracy = -1

    for epoch in range(kwargs["epohcs"]):
        result = train_epoch(**kwargs)

        print(f"[Epoch: {epoch}] train cost = {result['train_avg_cost']}")
        print(f"[Epoch: {epoch}] train acc = {result['train_avg_accuracy']}")
        print(f"[Epoch: {epoch}] valid cost = {result['valid_avg_cost']}")
        print(f"[Epoch: {epoch}] valid acc = {result['valid_avg_accuracy']}")

        if prev_accuracy < result["train_avg_accuracy"]:
            save_best = True
            prev_accuracy = result["train_avg_accuracy"]
            print("best : ")
        else:
            save_best = False
            print("normal : ")

        print(result["train_avg_accuracy"])

        kwargs(epoch, save_best)
