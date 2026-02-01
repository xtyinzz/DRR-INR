import os
from glob import glob
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from configs.configs import Config
from args import parse_args
from util.utils import set_random_seed
from util.eval_util import post_train_eval_ddp
from datasets.sf_dataset import HDFieldDataset, HDFieldDatasetEval, eval_online_downsample
from models.srns import HDINRBase
from models.utils import get_fwd_fn_test


@torch.no_grad()
def test_HDINR_DDP(
    args, dataset:HDFieldDataset, model:HDINRBase,
    accl:Accelerator, dim_scale_factor:float=1.0
):
    device = accl.device
    
    tb_logdir = os.path.join("logs", "tensorboard", args.testname, args.test_output_dir)
    logdir = os.path.join("logs", args.testname, "model")
    if accl.is_main_process:
        print('start HDINR-DDP Training')
        
    tb_writer = SummaryWriter(log_dir=tb_logdir)

    if accl.is_main_process:
        print('=' * 50)
        print('DATASET STATISTICS')
        print('=' * 50)
        print(f'Train Data Shape: {dataset.data.shape}')
        print(f'  - Min/Max/Mean: {dataset.data.min():.3f} / {dataset.data.max():.3f} / {dataset.data.mean():.3f}')
        print(f'Train Conditions Shape: {dataset.cond.shape}')
        print(f'  - Min/Max/Mean: {dataset.cond.min(0).values} / {dataset.cond.max(0).values} / {dataset.cond.mean(0)}')
        print(f'Test Data Shape: {dataset.data_unseen.shape}')
        print(f'  - Min/Max/Mean: {dataset.data_unseen.min():.3f} / {dataset.data_unseen.max():.3f} / {dataset.data_unseen.mean():.3f}')
        print(f'Test Conditions Shape: {dataset.cond_unseen.shape}')
        print(f'  - Min/Max/Mean: {dataset.cond_unseen.min(0).values} / {dataset.cond_unseen.max(0).values} / {dataset.cond_unseen.mean(0)}')
        print('=' * 50)

    dataset.cond_batch *= 4
    evalds_train = HDFieldDatasetEval(dataset, split_spatial='test', use_transform=False)
    evalds_train.set_data(dataset.cond, dataset.cond_idx, dataset.data)
    print(dataset.cond.shape, dataset.cond_idx.shape, dataset.data.shape, evalds_train.x.shape)
    dataloader = DataLoader(evalds_train, batch_size=1, shuffle=False, num_workers=args.num_workers)

    evalds_test = HDFieldDatasetEval(dataset, split_spatial='test', use_transform=False)
    evalds_test.set_data(dataset.cond_unseen, dataset.cond_idx_unseen, dataset.data_unseen)
    test_dataloader = DataLoader(evalds_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    has_unseen_cond = len(test_dataloader) > 0
    
    spatially_sparse = dim_scale_factor < 1.0

    print(f"Total number of batches in an epoch before DDP: {len(dataloader)}")
        
    model = model.to(device)
    fwd_fn_train, fwd_fn_test = get_fwd_fn_test(model)

    model, dataloader, test_dataloader= accl.prepare(
        model, dataloader, test_dataloader, device_placement=[True, False, False]
    )

    print(f"Total number of batches in an epoch after DDP: {len(dataloader)}")
    print(f"Process rank: {accl.process_index}, Is main process: {accl.is_main_process}")

    step = 0
    starting_epoch = 0
    ckpt_dirs = [f for f in sorted(glob(os.path.join(logdir, '*'))) if os.path.isdir(f)]
    ckpt_dirs = [ckpt_dirs[i] for i in args.test_ckpt_indices]
    
    
    if accl.is_main_process:
        output_dir = os.path.join("logs", args.testname, args.test_output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print('Output Direcotry created at', output_dir)
        print('Selected CKPTs are:', ckpt_dirs)
        df = pd.DataFrame(columns=[
            'ckpt_epoch',
            'cdTrain_spFull_psnr',
            'cdTest_spFull_psnr',
            "cdTrain_spTrain_psnr",
            "cdTest_spTrain_psnr",
            'cdTrain_spFull_ssim',
            'cdTest_spFull_ssim',
            "cdTrain_spTrain_ssim",
            "cdTest_spTrain_ssim"
        ])
        per_field_df = []
        per_field_keys = []
        exp_names = ['cdTest_spFull', 'cdTrain_spFull', 'cdTest_spTrain', 'cdTrain_spTrain']
        dump_dirs = [ os.path.join(output_dir, name) for name in exp_names]

        if_dumps = [
            len(indices) > 0 for indices in
            [args.dump_idx_cdTest_spFull, args.dump_idx_cdTrain_spFull, args.dump_idx_cdTest_spTrain, args.dump_idx_cdTrain_spTrain]
        ]
        if_any_dump = any(if_dumps)
        
        if_execute = [
            has_unseen_cond and (not args.if_dump_only or if_dumps[0]),
            (not args.if_dump_only or if_dumps[1]),
            has_unseen_cond and (not args.if_dump_only or if_dumps[2]),
            (not args.if_dump_only or if_dumps[3])
        ]

        for i, dump_dir in enumerate(dump_dirs):
            if if_execute[i]:
                os.makedirs(dump_dir, exist_ok=True)
        
    for i, ckpt_dir in enumerate(ckpt_dirs):
        accl.load_state(ckpt_dir)
        
        if accl.is_main_process:
            df.loc[i] = None
            df.loc[i, 'ckpt_epoch'] = args.test_ckpt_indices[i]
            print(f"Evaluating {args.test_ckpt_indices[i]}th CKPT.")
            
        if if_execute[0]:
            test_metrics, test_df = post_train_eval_ddp(
                args.test_ckpt_indices[i], accl, test_dataloader,
                model, fwd_fn_test, tb_writer, dataset=dataset,
                dump_idx=args.dump_idx_cdTest_spFull, dump_path=dump_dirs[0]
            )
            if accl.is_main_process:
                per_field_df.append(test_df)
                per_field_keys.append('cdTest_spFull')
                df.loc[i, 'cdTest_spFull_psnr'] = test_metrics['test_psnr']
                df.loc[i, 'cdTest_spFull_ssim'] = test_metrics['test_ssim']
                print(f"Evaluating Training Data {args.test_ckpt_indices[i]}th CKPT. Training:")
        else:
            print(f"No testing data found. Skipping testing.")
        
        
        if if_execute[1]:
            train_metrics, train_df = post_train_eval_ddp(
                args.test_ckpt_indices[i], accl, dataloader,
                model, fwd_fn_train, tb_writer, dataset=dataset,
                dump_idx=args.dump_idx_cdTrain_spFull, dump_path=dump_dirs[1]
            )
            per_field_df.append(train_df)
            per_field_keys.append('cdTrain_spFull')
            df.loc[i, 'cdTrain_spFull_psnr'] = train_metrics['test_psnr']
            df.loc[i, 'cdTrain_spFull_ssim'] = train_metrics['test_ssim']
        
        del evalds_train, evalds_test
        del dataloader, test_dataloader
        if spatially_sparse:
            print(f'**Spatially Sparse Dataset** Full coordinate evaluation done, Evaluating training resolution:')
            if if_execute[2]:
                print(f'Before downsampling, test data_unseen shape: {dataset.data_unseen.shape}')
                dataset.data_unseen = eval_online_downsample(dataset, dataset.data_unseen, scale_factor=dim_scale_factor, device=accl.device)
                dataset.dims = torch.tensor(dataset.data_unseen.shape[1:])
                print(f'After downsampling, test data_unseen shape: {dataset.data_unseen.shape}',
                      '\tmin:', dataset.data_unseen.min(), 'max:', dataset.data_unseen.max(), 'mean:', dataset.data_unseen.mean())

                evalds_test_sparseCoord = HDFieldDatasetEval(dataset, split_spatial='train', use_transform=False)
                evalds_test_sparseCoord.set_data(dataset.cond_unseen, dataset.cond_idx_unseen, dataset.data_unseen)
                test_dataloader_FC = DataLoader(evalds_test_sparseCoord, batch_size=1, shuffle=False, num_workers=args.num_workers)
                test_sparseCoord_metrics, test_sparseCoord_df = post_train_eval_ddp(
                    args.test_ckpt_indices[i], accl, test_dataloader_FC,
                    model, fwd_fn_test, tb_writer, dataset=dataset,
                    dump_idx=args.dump_idx_cdTest_spTrain, dump_path=dump_dirs[2]
                )
                if accl.is_main_process:
                    per_field_df.append(test_sparseCoord_df)
                    per_field_keys.append('cdTest_spTrain')
                    df.loc[i, 'cdTest_spTrain_psnr'] = test_sparseCoord_metrics['test_psnr']
                    df.loc[i, 'cdTest_spTrain_ssim'] = test_sparseCoord_metrics['test_ssim']
                    print(f"Complete test condition with FULL COORD on {args.test_ckpt_indices[i]}th CKPT. Next Training:")
            else:
                print('\t No unseen conditions for spatial generalization evaluation')
                
            print(f'Before downsampling, train data shape: {dataset.data.shape}')
            
            if if_execute[3]:
                dataset.data = eval_online_downsample(dataset, dataset.data, scale_factor=dim_scale_factor, device=accl.device)
                dataset.dims = torch.tensor(dataset.data.shape[1:])
                print(f'After downsampling, train data shape: {dataset.data.shape}',
                      '\tmin:', dataset.data.min(), 'max:', dataset.data.max(), 'mean:', dataset.data.mean())

                evalds_train_sparseCoord = HDFieldDatasetEval(dataset, split_spatial='train', use_transform=False)
                evalds_train_sparseCoord.set_data(dataset.cond, dataset.cond_idx, dataset.data)
                dataloader_FC = DataLoader(evalds_train_sparseCoord, batch_size=1, shuffle=False, num_workers=args.num_workers)
                train_sparseCoord_metrics, train_sparseCoord_df = post_train_eval_ddp(
                    args.test_ckpt_indices[i], accl, dataloader_FC,
                    model, fwd_fn_train, tb_writer, dataset=dataset,
                    dump_idx=args.dump_idx_cdTrain_spTrain, dump_path=dump_dirs[3]
                )
                if accl.is_main_process:
                    per_field_df.append(train_sparseCoord_df)
                    per_field_keys.append('cdTrain_spTrain')
                    df.loc[i, 'cdTrain_spTrain_psnr'] = train_sparseCoord_metrics['test_psnr']
                    df.loc[i, 'cdTrain_spTrain_ssim'] = train_sparseCoord_metrics['test_ssim']
            
        if accl.is_main_process:
            df.to_csv(os.path.join(output_dir, 'metrics.csv'))
            print(f"Metrics saved {os.path.join(output_dir, f'metrics.csv')}")
            per_field_df = pd.concat(per_field_df, keys=per_field_keys, names=['split', 'field_idx'])
            per_field_path = os.path.join(output_dir, f'per_field_metrics_{args.test_ckpt_indices[i]:03}.csv')
            per_field_df.to_csv(per_field_path)
            print(f"Per field metrics saved {per_field_path}")
            

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    set_random_seed(args.random_seed)

    # Load configuration file
    with open(args.config, 'r') as f:
        config_list = yaml.load(f, yaml.FullLoader)

    # Select configuration ID
    args.config_id = args.config_id if args.config_id != -1 else 0
    if isinstance(args.config_id, list):
        args.config_id = args.config_id[0]

    print(f"Running config_id {args.config_id}")

    # Initialize configuration and model
    config = Config(config_list[args.config_id])
    config.set_args(args)
    model = config.get_model()

    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize accelerator
    accl = Accelerator()
    config.config['dataset']['param']['device'] = accl.device

    # Set dimension scale factor to let full resolution evaluation be done first
    dim_scale_factor = config.config['dataset']['param'].get('dim_scale_factor', 1)
    config.config['dataset']['param']['dim_scale_factor'] = 1

    dataset = config.get_dataset()
    dataset._load_data_unseen()

    test_HDINR_DDP(args, dataset, model, accl, dim_scale_factor)

