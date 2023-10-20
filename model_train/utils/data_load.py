from torchvision import datasets, transforms


def data_load(data_path: str):
    return datasets.MNIST(
        root=data_path,  # 다운로드 경로 지정
        train=True,  # False를 지정하면 테스트 데이터로 다운로드
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        ),
        download=True,
    )
