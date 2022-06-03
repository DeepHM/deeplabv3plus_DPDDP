import os
from tqdm import tqdm
import json
import time
import datetime
import argparse
from typing import Tuple
import inspect
import math
import logging

import numpy as np
from utils import losses
from utils import Logger, helpers
from utils.torchsummary import summary
import utils.lr_scheduler
from utils.metrics import eval_metrics, AverageMeter

import dataloaders
from dataloaders.voc import VOCAugDataset, VOCDataset

from trainer import Trainer
import models

import torch
from torch.utils import tensorboard
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])
def reset_metrics():
    global batch_time, data_time, total_loss, total_inter, total_union, total_correct, total_label
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0
    return batch_time,data_time,total_loss,total_inter,total_union,total_correct,total_label
def update_seg_metrics(correct, labeled, inter, union):
    global total_correct, total_label, total_inter, total_inter, total_union

    total_correct += correct
    total_label += labeled
    total_inter += inter
    total_union += union
def get_seg_metrics() :
    global num_classes, total_correct, total_label, total_inter, total_union
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    return {
        "Pixel_Accuracy": np.round(pixAcc, 3),
        "Mean_IoU": np.round(mIoU, 3),
        "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 3)))
    }
def save_checkpoint(checkpoint_dir,base_logger, model,optimizer,config, epoch, save_best=False) :
    state = {
        'arch': type(model).__name__,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config
    }
    filename = os.path.join(checkpoint_dir, f'epoch{epoch}.pth')
    base_logger.info(f'\nSaving a checkpoint: {filename} ...') 
    torch.save(state, filename)

    if save_best:
        filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
        torch.save(state, filename)
        base_logger.info("Saving current best: best_model.pth")    
    
    
def create_data_loaders(config:dict,
                        rank: int,
                        world_size: int) -> Tuple[DataLoader, DataLoader]:
    
    MEAN = [0.45734706, 0.43338275, 0.40058118]
    STD = [0.23965294, 0.23532275, 0.2398498]
    default_kwargs = {'data_dir':None, 'batch_size':None, 'split':None,
              'crop_size':None, 'base_size':None, 'scale':True, 'num_workers':1, 'val':False,
              'shuffle':False, 'flip':False, 'rotate':False, 'blur': False, 'augment':False, 'val_split': None, 'return_id':False,
              'dist_sampler':False}        
    kwargs = {
        'root': default_kwargs['data_dir'],
        'split': default_kwargs['split'],
        'mean': MEAN,
        'std': STD,
        'augment': default_kwargs['augment'],
        'crop_size': default_kwargs['crop_size'],
        'base_size': default_kwargs['base_size'],
        'scale': default_kwargs['scale'],
        'flip': default_kwargs['flip'],
        'blur': default_kwargs['blur'],
        'rotate': default_kwargs['rotate'],
        'return_id': default_kwargs['return_id'],
        'val': default_kwargs['val']
    }
    
    # Dataset
    train_kwargs = {**kwargs,**config['train_loader']['args']}
    train_kwargs['root'] = config['train_loader']['args']['data_dir']
    train_kwargs = {i:train_kwargs[i] for i in train_kwargs if i in kwargs.keys()}
    valid_kwargs = {**kwargs,**config['val_loader']['args']}
    valid_kwargs['root'] = config['val_loader']['args']['data_dir']
    valid_kwargs = {i:valid_kwargs[i] for i in valid_kwargs if i in kwargs.keys()}

    train_dataset = VOCAugDataset(**train_kwargs)
    valid_dataset = VOCDataset(**valid_kwargs)
    
    # DataLoader
    sampler = DistributedSampler(train_dataset,
                                 num_replicas=world_size,  # Number of GPUs
                                 rank=rank,  # GPU where process is running
                                 shuffle=True,  # Shuffling is done by Sampler
                                 seed=42)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config['train_loader']['args']['batch_size'],
                              shuffle=False,  # This is mandatory to set this to False here, shuffling is done by Sampler
                              num_workers=config['train_loader']['args']['num_workers'],
                              sampler=sampler,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                         batch_size=config['val_loader']['args']['batch_size'],
                         shuffle=True,
                         num_workers=config['val_loader']['args']['num_workers'],
                         pin_memory=True)
    
    return train_loader, valid_loader


def main(config:dict,
         rank: int,
         ) -> nn.Module:
    
    start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
    base_logger = logging.getLogger('base_log')
    train_logger = Logger()
    
    # CHECKPOINTS & TENSOBOARD
    checkpoint_dir = os.path.join(config['trainer']['save_dir'], config['name'], start_time)
    helpers.dir_exists(checkpoint_dir)
    config_save_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_save_path, 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)
    writer_dir = os.path.join(config['trainer']['log_dir'], config['name'], start_time)
    writer = tensorboard.SummaryWriter(writer_dir)
    
    # Data Loaders
    train_loader, valid_loader= create_data_loaders(config, rank, world_size)    
    
    # Model
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    
    # Optimizer
    if config['optimizer']['differential_lr']:
        trainable_params = [{'params': filter(lambda p:p.requires_grad, model.module.get_decoder_params())},
                            {'params': filter(lambda p:p.requires_grad, model.module.get_backbone_params()), 
                            'lr': config['optimizer']['args']['lr'] / 10}]
    else:
        trainable_params = filter(lambda p:p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    
    # Loss and LR_scheduler
    start_epoch, end_epoch = 0, config['trainer']['epochs']
    loss_fn = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])
    lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler']['type'])(optimizer, end_epoch, len(train_loader))
    
    # Train
    global num_classes
    num_classes = train_loader.dataset.num_classes
    log_step = config['trainer'].get('log_per_iter', int(np.sqrt(train_loader.batch_size)))
    wrt_mode = 'train'
    
    for epoch in range(start_epoch, end_epoch+1) :
        base_logger.info('\n')
        batch_start_time = time.time()
        
        model.train()
        train_loader.sampler.set_epoch(epoch)
        
        if config['arch']['args']['freeze_bn'] :
            model.freeze_bn()
            
#         batch_time, data_time, total_loss, total_inter, total_union, total_correct, total_label = reset_metrics()
        reset_metrics()
        tbar = tqdm(train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            data_time.update(time.time() - batch_start_time)
            lr_scheduler.step(epoch=epoch-1)
            
            optimizer.zero_grad()
            output = model(data)
            
            if config['arch']['type'][:3] == 'PSP':
                assert output[0].size()[2:] == target.size()[1:]
                assert output[0].size()[1] == num_classes 
                loss = loss_fn(output[0], target)
                loss += loss_fn(output[1], target) * 0.4
                output = output[0]
            else:
                assert output.size()[2:] == target.size()[1:]
                assert output.size()[1] == num_classes 
                loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss.update(loss.item())
            
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            # LOGGING & TENSORBOARD
            if batch_idx % log_step == 0:
                wrt_step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar(f'{wrt_mode}/loss', loss.item(), wrt_step)
            
            # Train Evaluation
            seg_metrics = eval_metrics(output, target, num_classes)
            update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = get_seg_metrics().values()
            
            # PRINT INFO
            tbar.set_description('TRAIN ({}/{}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                                                epoch+1,end_epoch, total_loss.average, 
                                                pixAcc, mIoU,
                                                batch_time.average, data_time.average))
            

        # METRICS TO TENSORBOARD
        seg_metrics  = get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]: 
            writer.add_scalar(f'{wrt_mode}/{k}', v, wrt_step)
        for i, opt_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], wrt_step)
            #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)
            
       # RETURN LOSS & METRICS
        train_results = {'loss': total_loss.average,
                **seg_metrics}     
        base_logger.info(f'\n         ## Info for epoch {epoch} ## ')    
        for k, v in train_results.items():
            base_logger.info(f'         {str(k):15s}: {v}')  
        if train_logger is not None:
            log = {'epoch' : epoch, **train_results}
            train_logger.add_entry(log)     
            
        # SAVE CHECKPOINT
        if (rank == 0) and (epoch % config['trainer']['save_period'] == 0) :
            print('Save !')
            save_checkpoint(checkpoint_dir,base_logger, model,optimizer,config, epoch, save_best=False)
            
            


    
    
# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        
    rank = args.local_rank
    world_size = config['n_gpu']
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=Backend.NCCL,
                                         init_method='env://')
        
    main(config=config,rank=rank)
    
# python -m torch.distributed.launch --nproc_per_node=4 ddp_tutorial_multi_gpu.py

    
    
    
    
    