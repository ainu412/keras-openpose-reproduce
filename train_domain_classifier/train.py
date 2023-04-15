import torch
import numpy as np
import random
from pathlib import Path
import cv2
import os
from model import INet
from dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from tqdm import tqdm
from utils import AverageMeter
import time
import math
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    parser = argparse.ArgumentParser('Model')

    parser.add_argument('--nnodes', type=int, default=1, help='number of computers')
    parser.add_argument('--node_rank', default=0, type=int, help='ranking of this computer')
    parser.add_argument('--nproc_per_node', default=2, type=int, help='number of gpus per node')
    parser.add_argument('--intention_keyword', default='right', help='filter single intention to train')

    args = parser.parse_args()
    args.world_size = args.nproc_per_node * args.nnodes
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,5'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(main_worker, nprocs=args.nproc_per_node, args=(args,))


def main_worker(local_rank, args):
    rank = args.node_rank * args.nproc_per_node + local_rank
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        world_size=args.world_size,
        rank=rank)

    setup_seed(0 + local_rank)
    torch.cuda.set_device(local_rank)

    timestr = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    # exp_dir = exp_dir.joinpath('')

    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    plot_dir = exp_dir.joinpath('tensorboard/')
    plot_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("INet")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/INet.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(str):
        logger.info(str)
        print(str)

    writer = SummaryWriter(str(plot_dir))

    # train_dataset = Dataset(args.intention_keyword)
    train_dataset = Dataset()
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        # shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        sampler=train_sampler
    )
    log_string("The number of training data is: %d" % len(train_dataset))

    model = INet().cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )

    num_epochs = 100
    for epoch in range(num_epochs):
        log_string('Epoch %d (%d/%s):' % (epoch, epoch, num_epochs))
        train_sampler.set_epoch(epoch)
        running_loss = AverageMeter()
        lr = 1e-3
        if 30 <= epoch < 60:
            lr = 1e-4
        elif 60 <= epoch < 100:
            lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()

        iterator = tqdm(train_loader, total=len(train_loader))
        for i, (img, label) in enumerate(iterator):
            # img, intention, label = img.cuda(), intention.cuda(), label.cuda()
            img, label = img.cuda(), label.cuda().float()
            out = model(img)
            loss = criterion(out, label).float()

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.world_size)
            running_loss.update(reduced_loss.item())

            iterator.set_postfix({
                'train loss': '{:.6f}'.format(running_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log_string('Train loss: %.6f' % running_loss.avg)
        writer.add_scalar("Train/Loss", running_loss.avg, epoch)

        if local_rank == 0:
            if (epoch + 1) % 1 == 0:
                savepath = str(checkpoints_dir) + '/model_{epoch+1:04d}.pth'
                log_string('saving model')
                state = {
                    'epoch': epoch,
                    'train_loss': running_loss.avg,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                print('model saved at ' + savepath)
                torch.save(state, savepath)


if __name__ == '__main__':
    main()
