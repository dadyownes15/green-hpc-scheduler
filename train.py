import os
import configparser

from src.training import Train
from src.utils import get_config_as_dict

WORKLOAD_PATH = "data/workloads/training_workload.swf"


def main() -> None:
    config = configparser.ConfigParser()
    config_path = os.path.join(os.getcwd(), "config_file", "config.ini")
    config.read(config_path)

    config_dict = get_config_as_dict(config)
    print(config_dict)

    train = Train(
        config_dict=config_dict,
        workload_path=WORKLOAD_PATH,
        save_freq=int(config_dict["n_steps"]),
    )
    train.run(save_checkpoints=True)


if __name__ == "__main__":
    main()
