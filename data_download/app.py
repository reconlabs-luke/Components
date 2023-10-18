import argparse


def main(args):
    print(args)


if __name__ == "__main__":
    print("data download")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        dest="path",
        default="./",
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
