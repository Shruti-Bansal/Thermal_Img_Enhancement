# from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime
from tensorboardX import SummaryWriter
from Datasets import __datasets__
from Model import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

cudnn.benchmark = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

parser = argparse.ArgumentParser(
    description="Unet for Image Enhancement"
)
parser.add_argument(
    "--model",
    default="UNet",
    help="select a model structure",
    choices=__models__.keys(),
)


parser.add_argument("--dataset", default="ms2", help="dataset name", choices=__datasets__.keys())
parser.add_argument("--ms2_datapath", default="/home/sara/Thermal_Img_Enhancement", help="data path")
parser.add_argument("--process", default="minmax", help="data preprocess mode (raw, minmax, hist_99)")
parser.add_argument("--resolution", default="640x256", help="image resolution (width x height)")
parser.add_argument(
    "--blur", default="", help="random gaussian blur to training data (kernal_size:probability). e.g. 3:0.5"
)

parser.add_argument("--lr", type=float, default=0.001, help="base learning rate")
parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
parser.add_argument("--test_batch_size", type=int, default=4, help="testing batch size")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
parser.add_argument(
    "--lrepochs",
    default="300:10",
    type=str,
    help="the epochs to decay lr: the downscale rate",
)

parser.add_argument(
    "--logdir",
    default="./checkpoints",
    help="the directory to save logs and checkpoints",
)
parser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
parser.add_argument("--loadckpt", default="", help="load the weights from a specific checkpoint")
parser.add_argument("--resume", action="store_true", help="continue training the model")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument("--summary_freq", type=int, default=20, help="the frequency of saving summary")
parser.add_argument("--save_freq", type=int, default=1, help="the frequency of saving checkpoint")
parser.add_argument("--wandb", action="store_true", help="use wandb to log instead of tensorboard")
parser.add_argument("--WANDB_API_KEY", type=str, default="1f89a8360882aeadbaeb5affc82d6bcffaf671a4") #need this

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

logdir = get_logdir_name(args.logdir, args.resume, datetime.now())
os.makedirs(logdir, exist_ok=True)
run_name = logdir.split("/")[-1]

print("args:", vars(args))
with open(os.path.join(logdir, "args.txt"), "a") as f:
    f.write(str(vars(args)))
    f.write("\n")

if args.wandb:
    # wandb init
    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = args.WANDB_API_KEY
    if args.resume:
        wandb.init(
            project="thermal-img-enhnc",
            entity="cmu-thermal",
            name=run_name,
            config=vars(args),
            resume="allow",
            dir=logdir,
        )
    else:
        wandb.init(
            project="thermal-img-enhnc", entity="cmu-thermal", name=run_name, config=vars(args), dir=logdir
        )
else:
    # tensorboard init
    print(f"creating new summary file at {logdir}")
    logger = SummaryWriter(logdir)

filepath = os.path.dirname(__file__)
train_file = os.path.join(filepath, "filenames/ms2_train.txt")
test_file = os.path.join(filepath, "filenames/ms2_test.txt")
results_path = os.path.join(filepath, "Results")

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(
    root=args.ms2_datapath,
    datalist=train_file,
    data_split="train",
    process=args.process,
    resolution=args.resolution, #check resolution
    sampling_step=1,
    set_length=1,
    set_interval=1,
    blur=args.blur,
)

test_dataset = StereoDataset(
    root=args.ms2_datapath,
    datalist=test_file,
    data_split="val",
    process=args.process,
    resolution=args.resolution,
    sampling_step=1,
    set_length=1,
    set_interval=1,
)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
TestImgLoader = DataLoader(
    test_dataset, args.test_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False
)

# model, optimizer
model = __models__[args.model]()
#model = nn.DataParallel(model)
model.cuda()
if args.wandb:
    wandb.watch(model, log_freq=10)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)
# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    start_epoch = state_dict["epoch"] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict["model"].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))





def train():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            #print("**Batch:",batch_idx)
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample) #, compute_metrics=do_summary)
            if do_summary and not args.wandb:
                save_scalars(logger, "train", scalar_outputs, global_step)
                save_images(logger, "train", image_outputs, global_step)
            if do_summary and args.wandb:
                wandb_scalars("train", scalar_outputs, global_step)
                wandb_images("train", image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                "Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}".format(
                    epoch_idx,
                    args.epochs,
                    batch_idx,
                    len(TrainImgLoader),
                    loss,
                    time.time() - start_time,
                )
            )

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                checkpoint_data,
                "{}/checkpoint_{:0>6}.ckpt".format(logdir, epoch_idx),
            )
            if args.wandb:
                wandb.save("{}/checkpoint_{:0>6}.ckpt".format(logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        iter = 0
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            # do_summary = global_step % args.summary_freq == 0
            do_summary = global_step % 1 == 0
            loss, scalar_outputs, image_outputs = test_sample(sample) #, compute_metrics=do_summary)
            if do_summary and not args.wandb:
                save_scalars(logger, "test", scalar_outputs, global_step)
                #save_images(logger, "test", image_outputs, global_step)
            if do_summary and args.wandb:
                wandb_scalars("test", scalar_outputs, global_step)
                #wandb_images("test", image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print(
                "Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}".format(
                    epoch_idx,
                    args.epochs,
                    batch_idx,
                    len(TestImgLoader),
                    loss,
                    time.time() - start_time,
                )
            )

            test_images = image_outputs["enhn_img"]
            test_images = torch.permute(test_images,(0, 2, 3, 1))

            
            if epoch_idx == 99:
                
                path_str = '/home/sara/Thermal_Img_Enhancement/Results/Img'
                for i in range(args.test_batch_size):
                    test_img = test_images[i, :, :, :]
                    
                    test_img_np = test_img.detach().cpu().numpy()
                    test_img_np = np.squeeze(test_img_np, axis=2)
                    path_img = path_str + str(iter) + '.png'
                    cv2.imwrite(path_img, test_img_np)
                    iter = iter + 1

            del image_outputs

        avg_test_scalars = avg_test_scalars.mean()
        if not args.wandb:
            save_scalars(logger, "fulltest", avg_test_scalars, epoch_idx)
        if args.wandb:
            scalar_dict = {f"fulltest/{key}": value for key, value in avg_test_scalars.items()}
            scalar_dict = {k: v[0] if isinstance(v, list) and len(v) > 0 else v for k, v in scalar_dict.items()}
            scalar_dict["epoch"] = epoch_idx
            wandb.log(scalar_dict)
        print("avg_test_scalars", avg_test_scalars)
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    img_gray, img_thr = (
        sample["gray"],
        sample["thr"],
    )
    img_gray = img_gray.cuda()
    
    img_thr = img_thr.cuda()
    img_thr = torch.permute(img_thr,(0, 3, 1, 2))
    img_gray = torch.permute(img_gray, (0, 3, 1, 2))
    #print(img_thr.shape)
    
    optimizer.zero_grad()
    img_gen = model(img_thr)

    loss_fn = nn.L1Loss()
    loss =  loss_fn(img_gray, img_gen)
    scalar_outputs = {"loss": loss}
    image_outputs = {
        "enhn_img": img_gen,
    }
    
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
 
def test_sample(sample, compute_metrics=True):
    model.eval()
    img_gray, img_thr = (
        sample["gray"],
        sample["thr"],
    )
    img_gray = img_gray.cuda()
    img_thr = img_thr.cuda()
    img_thr = torch.permute(img_thr,(0, 3, 1, 2))
    img_gray = torch.permute(img_gray, (0, 3, 1, 2))

    optimizer.zero_grad()
    img_gen = model(img_thr)

    #print("Img_Gen:", img_gen.shape)
    loss_fn = nn.L1Loss()
    loss =  loss_fn(img_gray, img_gen)
    scalar_outputs = {"loss": loss}
    image_outputs = {
        "enhn_img": img_gen,
    }
    

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == "__main__":
    train()
    #wandb.finish()