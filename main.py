import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from runners.diffusion import Diffusion
from config import ARGUMENTS, conditions

torch.set_printoptions(sci_mode=False)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    # Adding all arguments here, using the static dictionary ARGUMENTS
    parser.add_argument("--seed", type=int, default=ARGUMENTS["seed"], help="Random seed")
    parser.add_argument("--exp", type=str, default=ARGUMENTS["exp"], help="Path for saving running related data.")
    
    # Set --doc to use the value from ARGUMENTS, and remove the required=True part
    parser.add_argument("--doc", type=str, default=ARGUMENTS["doc"], help="A string for documentation purpose.")
    
    parser.add_argument("--comment", type=str, default=ARGUMENTS["comment"], help="A string for experiment comment")
    parser.add_argument("--verbose", type=str, default=ARGUMENTS["verbose"], help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--sample", action="store_true", help="Whether to produce samples from the model")
    parser.add_argument("-i", "--image_folder", type=str, default=ARGUMENTS["image_folder"], help="The folder name of samples")
    parser.add_argument("--ni", type=bool, default=ARGUMENTS["ni"], help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--timesteps", type=int, default=ARGUMENTS["timesteps"], help="Number of steps involved")
    parser.add_argument("--model_type", type=str, default=ARGUMENTS["model_type"], help="face | imagenet")
    parser.add_argument("--batch_size", type=int, default=ARGUMENTS["batch_size"])
    parser.add_argument("--class_num", type=int, default=ARGUMENTS["class_num"])
    parser.add_argument("-s", "--sample_strategy", type=str, default=ARGUMENTS["sample_strategy"])
    parser.add_argument("--mu", type=float, default=ARGUMENTS["mu"])
    parser.add_argument("--rho_scale", type=float, default=ARGUMENTS["rho_scale"])
    parser.add_argument("--prompt", type=str, default=ARGUMENTS["prompt"])
    parser.add_argument("--stop", type=int, default=ARGUMENTS["stop"])
    parser.add_argument("--ref_path", type=str, default=ARGUMENTS["ref_path"])
    parser.add_argument("--ref_path2", type=str, default=ARGUMENTS["ref_path2"])
    parser.add_argument("--scale_weight", type=float, default=ARGUMENTS["scale_weight"])
    parser.add_argument("--rt", type=int, default=ARGUMENTS["rt"])

    # Parse arguments as usual
    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args


def main_multi_conditions():
    args = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    print(f"main_multi_conditions(): conditions = {conditions}")

    try:
        runner = Diffusion(args)
        runner.sample_multi_conditions(conditions)
    except Exception:
        logging.error(traceback.format_exc())

    return 0

if __name__ == "__main__":
    sys.exit(main_multi_conditions())