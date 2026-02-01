import os
import time
from glob import glob
from re import search as re_search

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from args import parse_args
from configs.configs import Config
from datasets.sf_dataset import HDFieldDataset
from models.utils import get_fwd_fn
from util.eval_util import post_train_eval_ddp_inr, post_train_report
from util.utils import save_checkpoint_ddp, set_random_seed



def train_INR_DDP(
    args, config, dataset: HDFieldDataset, model: torch.nn.Module,
    accl: Accelerator, optimizer=None, scheduler=None
):
    """
    Trains a standard Implicit Neural Representation (INR) model with Accelerate,
    supporting Distributed Data Parallel (DDP).

    This function handles the training loop for a single INR, which fits one data sample.
    It leverages the Accelerate library for DDP training.
    """
    device = accl.device
    max_epochs = args.maxsteps
    model = model.to(device)

    if optimizer is None:
        optimizer = config.get_optim(model)

    # For standard INR, we fit one sample at a time. Batch size is 1.
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # No need to shuffle a single-item dataset
        num_workers=args.num_workers
    )

    if not args.disable_scheduler and scheduler is None:
        total_steps = max_epochs * len(dataloader)
        eta_min = config.config['scheduler']['param']['eta_min']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=eta_min
        )

    model, optimizer, dataloader, scheduler = accl.prepare(
        model, optimizer, dataloader, scheduler
    )

    tb_writer = None
    logdir = os.path.join("logs", args.testname, "model")
    if accl.is_main_process:
        accl.print('Starting INR-DDP Training')
        tb_logdir = os.path.join("logs", "tensorboard", args.testname)
        tb_writer = SummaryWriter(log_dir=tb_logdir)
        if args.load_from is None:
            os.makedirs(logdir, exist_ok=True)
            # Consider safer cleanup methods if necessary
            os.system(f"rm -rf {tb_logdir}/*")
            os.system(f"rm -rf {logdir}/*")

    model_wrapper = accl.unwrap_model(model)
    model_loss_fn = model_wrapper.get_loss

    step = 0
    starting_epoch = 0
    if args.load_from is not None:
        load_path = args.load_from
        if load_path == "":
            # Find the most recent checkpoint directory
            dirs = sorted([
                p for p in glob(os.path.join(logdir, '*')) if os.path.isdir(p)
            ], key=os.path.getmtime)
            if dirs:
                load_path = dirs[-1]

        if load_path and os.path.exists(load_path):
            accl.print(f"Resuming from checkpoint: {load_path}")
            accl.load_state(load_path)
            try:
                load_dir_name = os.path.basename(load_path)
                starting_epoch = int(re_search(r'ep(\d+)', load_dir_name).group(1)) + 1
                step = starting_epoch * len(dataloader)
            except (AttributeError, ValueError):
                accl.print("Could not parse epoch from checkpoint directory name. Starting from epoch 0.")
        else:
            accl.print("No checkpoint found to resume from. Starting from scratch.")

    tic = time.time()
    for ei in range(starting_epoch, max_epochs):
        model.train()
        for x, y in dataloader:
            # Squeeze batch dimension of 1 for standard INR training
            x, y = x.squeeze(0), y.squeeze(0)
            with accl.autocast():
                pred = model(x)
                if not isinstance(pred, tuple):
                    pred = pred.view(*y.shape)
                loss_list, loss = model_loss_fn(pred, y)
                if isinstance(pred, tuple):
                    pred = pred[0].view(*y.shape)

            optimizer.zero_grad()
            accl.backward(loss)
            optimizer.step()
            if scheduler:
                scheduler.step()

            if accl.is_main_process:
                if step % args.reportsteps == 0 and step > 0:
                    post_train_report(step, pred, y, loss, loss_list, tb_writer, tic, ei)
            step += 1

        if ei % args.teststeps == 0 and accl.is_main_process:
            model.eval()
            metrics = {'loss': loss.cpu().item()}
            save_checkpoint_ddp(accl, logdir, epoch=ei, metrics=metrics)

    accl.wait_for_everyone()
    if accl.is_main_process:
        eval_dir = os.path.join("logs", args.testname, args.test_output_dir)
        os.makedirs(eval_dir, exist_ok=True)
        post_train_eval_ddp_inr(
            ei, accl, dataset, model, tb_writer,
            args.eval_batch_size, args=args, dump_path=eval_dir
        )
        if tb_writer:
            tb_writer.close()

    accl.end_training()


def train_HDINR_DDP(
    args, config, dataset: HDFieldDataset, model: torch.nn.Module,
    accl: Accelerator, optimizer=None, scheduler=None
):
    """
    Trains a multi-field High-Dimensional Implicit Neural Representation (HDINR) using
    Distributed Data Parallel (DDP) with Accelerate.
    """
    device = accl.device
    max_epochs = args.maxsteps
    model = model.to(device)

    if optimizer is None:
        optimizer = config.get_optim(model)

    # Setup DataLoader with optional importance sampling
    sampler = None
    shuffle = True
    if dataset.impsmp_conds is not None:
        accl.print("Using WeightedRandomSampler for importance sampling.")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=dataset.impsmp_conds,
            num_samples=len(dataset),
            replacement=True
        )
        shuffle = False  # Sampler handles shuffling

    dataloader = DataLoader(
        dataset,
        batch_size=dataset.cond_batch,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers
    )

    if not args.disable_scheduler and scheduler is None:
        total_steps = max_epochs * len(dataloader)
        eta_min = config.config['scheduler']['param']['eta_min']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=eta_min
        )

    # Prepare all components with Accelerate
    model, optimizer, dataloader, scheduler = accl.prepare(
        model, optimizer, dataloader, scheduler
    )

    tb_writer = None
    logdir = os.path.join("logs", args.testname, "model")
    if accl.is_main_process:
        accl.print('Starting HDINR-DDP Training')
        tb_logdir = os.path.join("logs", "tensorboard", args.testname)
        tb_writer = SummaryWriter(log_dir=tb_logdir)
        if args.load_from is None:
            os.makedirs(logdir, exist_ok=True)
            # Clean up previous logs and checkpoints for a fresh run
            os.system(f"rm -rf {tb_logdir}/*")
            os.system(f"rm -rf {logdir}/*")

    model_wrapper = accl.unwrap_model(model)
    model_loss_fn = model_wrapper.get_loss
    fwd_fn = get_fwd_fn(model_wrapper)

    step = 0
    starting_epoch = 0
    if args.load_from is not None:
        load_path = args.load_from
        if load_path == "":
            # Find the most recent checkpoint directory
            dirs = sorted([
                p for p in glob(os.path.join(logdir, '*')) if os.path.isdir(p)
            ], key=os.path.getmtime)
            if dirs:
                load_path = dirs[-1]

        if load_path and os.path.exists(load_path):
            accl.print(f"Resuming from checkpoint: {load_path}")
            accl.load_state(load_path)
            try:
                load_dir_name = os.path.basename(load_path)
                starting_epoch = int(re_search(r'ep(\d+)', load_dir_name).group(1)) + 1
                step = starting_epoch * len(dataloader)
            except (AttributeError, ValueError):
                accl.print("Could not parse epoch from checkpoint name. Starting from epoch 0.")
        else:
            accl.print("No checkpoint found to resume from. Starting from scratch.")

    tic = time.time()
    for ei in range(starting_epoch, max_epochs):
        model.train()
        for x, cond, cond_i, y in dataloader:
            with accl.autocast():
                pred = fwd_fn(model, (x, cond, cond_i))
                if not isinstance(pred, tuple):
                    pred = pred.view(*y.shape)
                loss_list, loss = model_loss_fn(pred, y)
                if isinstance(pred, tuple):
                    pred = pred[0].view(*y.shape)

            optimizer.zero_grad()
            accl.backward(loss)
            optimizer.step()
            if scheduler:
                scheduler.step()

            if accl.is_main_process:
                if step % args.reportsteps == 0 and step > 0:
                    post_train_report(step, pred, y, loss, loss_list, tb_writer, tic, ei)
            step += 1

        if ei % args.teststeps == 0 and accl.is_main_process:
            model.eval()
            metrics = {'loss': loss.cpu().item()}
            save_checkpoint_ddp(accl, logdir, epoch=ei, metrics=metrics)

    accl.wait_for_everyone()
    if accl.is_main_process:
        model.eval()
        metrics = {'loss': loss.cpu().item()}
        save_checkpoint_ddp(accl, logdir, epoch=ei, metrics=metrics)
        if tb_writer:
            tb_writer.close()

    accl.end_training()

def main():
    """
    Main function to parse arguments, load configuration, and start the training process.
    """
    parser = parse_args()
    args = parser.parse_args()

    set_random_seed(args.random_seed)

    # Load YAML configuration file
    with open(args.config, 'r') as f:
        config_list = yaml.load(f, yaml.FullLoader)

    # Select the specified configuration
    config_id = args.config_id[0] if isinstance(args.config_id, list) else args.config_id
    if config_id == -1:
        config_id = 0
    print(f"Running with config_id: {config_id}")

    config = Config(config_list[config_id])
    config.set_args(args)

    # Initialize Accelerator for distributed training
    accl = Accelerator()

    # Initialize model and dataset
    model = config.get_model()
    dataset = config.get_dataset()

    # Select and run the appropriate training scheme
    if args.train_scheme == "hdinr_ddp":
        train_HDINR_DDP(args, config, dataset, model, accl)
    elif args.train_scheme == "inr_ddp":
        train_INR_DDP(args, config, dataset, model, accl)
    else:
        raise NotImplementedError(
            f"Training scheme '{args.train_scheme}' is not implemented."
        )

if __name__ == "__main__":
    main()
