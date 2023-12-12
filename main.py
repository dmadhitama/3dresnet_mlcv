from pathlib import Path
import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import SGD, lr_scheduler

from opts import parse_opts
from model import (generate_model, load_pretrained_model,
                   get_fine_tuning_parameters)
from mean import get_mean_std
from validation import val_epoch

from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)

from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)

from temporal_transforms import Compose as TemporalCompose
from dataset import get_training_data, get_validation_data, get_validation_data_multiclips
from training import train_epoch
from utils import Colors, print_color
from torch.utils.tensorboard import SummaryWriter

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.train_path = opt.root_path / opt.train_path
        opt.val_path = opt.root_path / opt.val_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path

        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    
    with (opt.result_path / 'opts.json').open('w') as opt_file:
        json.dump(vars(opt), opt_file, default=json_serial)

    return opt

def get_train_utils(opt, model_parameters):
    print_color("Get Train Dataset", Colors.YELLOW)
    assert opt.train_crop in ['random', 'corner', 'center']
    spatial_transform = []

    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
        
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
        
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    
    normalize = Normalize(opt.mean, opt.std)
    
    spatial_transform.append(ToTensor())
    
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    assert opt.train_t_crop in ['random', 'center']
    
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))

    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
        
    temporal_transform = TemporalCompose(temporal_transform)

    train_data = get_training_data(opt.train_path, opt.annotation_path,
                                   opt.dataset, spatial_transform, temporal_transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.n_threads,
                                               pin_memory=True)
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
        
    optimizer = SGD(model_parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)

    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones)

    return (train_loader, optimizer, scheduler)

def get_val_utils(opt):
    print_color("Get Validation Dataset", Colors.YELLOW)
    normalize = Normalize(opt.mean, opt.std)

    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor()
    ]

    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))

    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))

    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))

    temporal_transform = TemporalCompose(temporal_transform)

    val_data = get_validation_data(opt.val_path,
                                    opt.annotation_path, opt.dataset,
                                    spatial_transform,
                                    temporal_transform)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             )
    return val_loader

def get_val_utils_multiclips(opt):
    print_color("Get Multi Validation Dataset", Colors.RED)
    normalize = Normalize(opt.mean, opt.std)

    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor()
    ]

    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    
    temporal_transform.append(TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))
    
    temporal_transform = TemporalCompose(temporal_transform)
    val_data, collate_fn = get_validation_data_multiclips(opt.val_path,
                                    opt.annotation_path, opt.dataset,
                                    spatial_transform,
                                    temporal_transform)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size // opt.n_val_samples),
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=opt.n_threads,
                                             collate_fn=collate_fn)
    return val_loader

def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)

def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model

def main_worker(opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    loss_scale = 333.0
    
    model = generate_model(opt)

    if opt.pretrain_path:
        model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                      opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)

    if torch.cuda.device_count() > 1:
        print_color(f'Use {torch.cuda.device_count()} GPUs', Colors.RED)
        model = torch.nn.DataParallel(model)
    
    model.to(opt.device)

    if opt.pretrain_path:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
    else:
        parameters = model.parameters()

    class_weights = torch.ones(opt.n_classes) * loss_scale
    criterion = BCEWithLogitsLoss(weight=class_weights).to(opt.device)

    if not opt.no_train:
        (train_loader, optimizer, scheduler) = get_train_utils(opt, parameters)
    if not opt.no_val:
        val_loader = get_val_utils(opt)

    tb_writer = SummaryWriter(log_dir=opt.result_path)

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt.device, tb_writer)

            if i % opt.checkpoint == 0:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,scheduler)

        if not opt.no_val:
            val_loss = val_epoch(i, val_loader, model, criterion, opt.device, tb_writer)
        
        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()

if __name__ == '__main__':
    opt = get_opt()
    
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_worker(opt)