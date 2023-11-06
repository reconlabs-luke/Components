import torch
import numpy as np
import random
import argparse
import mlflow

from torchvision import datasets, transforms
from tqdm import tqdm


def main(args):
    seed = 2023
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    mnist_test = datasets.MNIST(
        root=args.data_path,  # 다운로드 경로 지정
        train=False,  # False를 지정하면 테스트 데이터로 다운로드
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        ),
        download=True,
    )

    test_loader = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # model load
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    model_uri = None
    if args.mlflow_run_id:
        client = mlflow.client.MlflowClient()
        artifact_uri = client.get_model_version_download_uri(
            args.mlflow_run_id,
        )
        print("from : ", artifact_uri)
        if artifact_uri.split("/")[-1] == args.model_artifact_path:
            model_uri = f"runs:/{args.mlflow_run_id}/{args.model_artifact_path}"
        else:
            print("Incorrect model_artifact_path : ", args.model_artifact_path)
    else:
        model_uri = f"models:/{args.model_name}/{args.model_stage}"

    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    accruacy = 0
    with mlflow.start_run(args.mlflow_run_id):
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(test_loader)):
                output = model(data)
                correct_pred = torch.argmax(output, 1) == target
                accruacy += correct_pred.float().mean() / len(test_loader)

            print(f"Accuracy : {accruacy.cpu().detach().numpy()}")

        mlflow.log_metric("accuracy", accruacy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--data-path",
        dest="data_path",
        type=str,
        help="/path/to/data",
        default="/tmp",
        action="store",
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        dest="mlflow_tracking_uri",
        type=str,
        default="http://localhost:5000",
        action="store",
    )

    parser.add_argument(
        "--model-name",
        dest="model_name",
        type=str,
        default="MNIST",
        action="store",
    )

    parser.add_argument(
        "--model-stage",
        dest="model_stage",
        type=str,
        default="production",
        action="store",
    )

    parser.add_argument(
        "--mlflow-run-id",
        dest="mlflow_run_id",
        type=str,
        default=None,
        action="store",
    )

    parser.add_argument(
        "--model-artifact-path",
        dest="model_artifact_path",
        type=str,
        default="model",
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
