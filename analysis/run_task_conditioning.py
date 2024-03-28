import argparse
import logging
import random


from train import load_from_config, combine_config_args, update_args
from train_helper import generate_replay

from evaluate import make_env_creator, make_agent_creator

CURRICULUM_FILE = "neurips23_evaluation/heldout_task_with_embedding.pkl"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Parse environment argument", add_help=False)
    parser.add_argument("eval_model_path", type=str, default=None, help="Path to model to evaluate")
    parser.add_argument(
        "-c", "--curriculum", type=str, default=CURRICULUM_FILE, help="Path to curriculum file"
    )
    parser.add_argument(
        "-t",
        "--task-to-assign",
        type=int,
        default=None,
        help="The index of the task to assign in the curriculum file",
    )
    parser.add_argument(
        "-r", "--repeat", type=int, default=1, help="Number of times to repeat the evaluation"
    )
    clean_parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_known_args()[0].__dict__

    # required args when using train.py's helper functions
    args["no_track"] = True
    args["no_recurrence"] = False
    args["vectorization"] = "serial"
    args["debug"] = False

    # Generate argparse menu from config
    config = load_from_config("neurips23_start_kit")  # dummy learner
    args = combine_config_args(parser, args, config)
    args = update_args(args, mode="replay")

    agent_creator = make_agent_creator()
    env_creator = make_env_creator(args["curriculum"], "pvp")

    # Generate replay
    for i in range(args.repeat):
        generate_replay(
            args,
            env_creator,
            agent_creator,
            stop_when_all_complete_task=False,
            seed=random.randint(10000000, 99999999),
        )
        print(f"Generated replay for task {i+1}/{args.repeat}...")

    print("Done!")
