import os
import torch
from datetime import datetime


def get_logdir_name(path, resume, time):
    if resume:
        directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        date_format = "%m%d_%H:%M:%S"
        date_objects = [datetime.strptime(d, date_format) for d in directories]
        most_recent_dir = max(zip(date_objects, directories))[1]
        return os.path.join(path, most_recent_dir)
    run_name = time.strftime("%m%d_%H:%M:%S")
    return os.path.join(path, run_name)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = args.local_rank
        args.dist_url = "env://"
        os.environ["LOCAL_SIZE"] = str(torch.cuda.device_count())
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
