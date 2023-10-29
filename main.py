"""

CLI - Command Line Interface

"""

import argparse

from e2.data_ops import DatasetIO

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="<Desc to be added>")
    args.add_argument(
        "--data_path", default="data.csv", type=str, help="path of the read data"
    )

    args.add_argument(
        "--save_path", default="data.csv", type=str, help="path for saving data"
    )

    data_path = args.parse_args().data_path
    save_path = args.parse_args().save_path

    dataset_io = DatasetIO(data_path=data_path)
    dataset = dataset_io.read_data()

    dataset_io.save_data(data=dataset, save_path=save_path)
