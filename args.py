import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='yaml config file for the network module (training, logs, etc.)'
    )
    parser.add_argument(
        "--config_id",
        type=int,
        nargs="+",
        default=-1,
        help="index/indices of config in the config_list to use",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="the GPU to use, None is for automatic device selection. \
             for multi-job YAML config, each job launched as \
             a new process with unique GPU",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_worker for dataloader",
    )
    parser.add_argument(
        "--train_scheme",
        type=str,
        default="default",
        help="training scheme to use",
    )
    parser.add_argument(
        "--reportsteps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--teststeps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--testname",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--maxsteps",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--load_from",
        type=str,
        default=None,
        help="Model path to load"
    )
    parser.add_argument(
        "--disable_scheduler",
        action="store_true",
        help="whether to use lr scheduler (can be on all or just backbone)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        help="Batch sizes for evaluation",
    )
    parser.add_argument(
        "--if_dump_only",
        action="store_true",
        help="whether to only inference on fields to be dumped",
    )
    parser.add_argument(
        "--dump_idx_cdTrain_spFull",
        type=int,
        nargs="+",
        default=[],
        help="Indices to dump field for cd_train space_full visualization",
    )
    parser.add_argument(
        "--dump_idx_cdTest_spFull",
        type=int,
        nargs="+",
        default=[],
        help="Indices to dump field for cd_test space_full visualization",
    )
    parser.add_argument(
        "--dump_idx_cdTest_spTrain",
        type=int,
        nargs="+",
        default=[],
        help="Indices to dump field for cd_test space_train visualization",
    )
    parser.add_argument(
        "--dump_idx_cdTrain_spTrain",
        type=int,
        nargs="+",
        default=[],
        help="Indices to dump field for cd_train space_train visualization",
    )
    parser.add_argument(
        "--test_output_dir",
        type=str,
        default=None,
        help="output directory for test results",
    )
    parser.add_argument(
        "--test_ckpt_indices",
        type=list[int],
        default=None,
        help="indices of checkpoints to test in args.testname/model",
    )
    # nerfacc args
    parser.add_argument(
        "--nerfacc_data_root",
        type=str,
        help="the root dir of the dataset",
    )
    parser.add_argument(
        "--nerfacc_train_split",
        type=str,
        default="train",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--nerfacc_scene",
        type=str,
        default="lego",
        help="which scene to use",
    )
    parser.add_argument(
        "--nerfacc_test_chunk_size",
        type=int,
        default=4096,
    )
    return parser