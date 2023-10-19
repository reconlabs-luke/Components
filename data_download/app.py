import argparse
from torchvision import datasets, transforms


def main(args):
    if args.path == "MNIST":
        datasets.MNIST(
            root=args.output_path,  # 다운로드 경로 지정
            train=True,  # False를 지정하면 테스트 데이터로 다운로드
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            ),
            download=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        dest="path",
        default="MNIST",
        action="store",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        default="/tmp",
        action="store",
    )

    args = parser.parse_args()
    main(args)
