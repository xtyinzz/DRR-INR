import os
from time import sleep
from pathlib import Path
import yaml
import subprocess
import shlex
from torch.cuda import device_count
from time import time
from args import parse_args

if __name__ == "__main__":
    
    time_start = time()
    
    parser = parse_args()
    parser.add_argument(
        '--use_accelerate',
        action='store_false'
    )
    parser.add_argument(
        '--accelerate_config',
        type=str,
        default='configs/hdinr/accl/single_node.yaml',
    )
    parser.add_argument(
        '--free_port',
        type=str
    )
    parser.add_argument(
        '--script',
        type=str,
        default='train_srn.py'
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_list = yaml.load(f, yaml.FullLoader)
    
    if args.config_id == -1:
        args.config_id = list(range(len(config_list)))
    config_list = [config_list[i] for i in args.config_id] # selecting the requested config ids
    print('config_ids', args.config_id)
    # prepare log dir for redirection of output
    for config in config_list:
        log_dir = Path("logs", config['train']['testname'])
        # os.system(f"rm -rf {log_dir}")
        os.makedirs(log_dir, exist_ok=True)
    
    avail_config_ids = args.config_id
    jobs = []
    opened_files = []
    avail_devices = [i for i in range(device_count())]
    while len(config_list) > 0:
        # issue jobs
        while len(avail_devices) > 0 and len(config_list) > 0:
            device = avail_devices.pop()
            config = config_list.pop()
            log_dir = Path("logs", config['train']['testname'])
            # os.makedirs(log_dir, exist_ok=True)
            # cmd = f"python renderer.py \
            #         --config={args.config} \
            #         --device={device} \
            #         --config_id={avail_config_ids.pop()} "
            # out_path = Path(log_dir, 'render_log.txt')

                    
            if args.use_accelerate:
                cmd = f"accelerate launch \
                        --config_file {args.accelerate_config} \
                        --main_process_port {args.free_port} \
                        {args.script} \
                        --config={args.config} \
                        --config_id={avail_config_ids.pop()}"
            else:
                cmd = f"python train_srn.py \
                        --config={args.config} \
                        --device={device} \
                        --config_id={avail_config_ids.pop()}"
                        
            script_name = os.path.splitext(os.path.basename(args.script))[0]
            
            out_path = Path(log_dir, f'{script_name}_log.txt')
            out_path = open(out_path, 'w+')
            opened_files.append(out_path)
            print(f"{time()-time_start:.2f}sec ***** issued a job ", shlex.split(cmd), "*** log_dir", log_dir, flush=True)
            job = subprocess.Popen(shlex.split(cmd), stdout=out_path, stderr=out_path)
            jobs.append({
                'job': job,
                'device': device
            })
        
        tmp_jobs = []
        # check if any job ends and release devices
        for job_dict in jobs:
            if job_dict['job'].poll() is not None:
                device = job_dict['device']
                avail_devices.append(device)
            else:
                # job not finished, keep it
                tmp_jobs.append(job_dict)
        jobs = tmp_jobs
        
        sleep(1)
        
    # all jobs issues, just need to wait for termination
    print("all jobs issued. Waiting for termination", flush=True)
    for job_dict in jobs:
        job_dict['job'].wait()
    print("all jobs terminated.", flush=True)
    
    for f in opened_files:
        f.close()
                
