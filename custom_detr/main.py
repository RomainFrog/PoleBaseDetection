"""
This is the main file for the custom DETR model.
It will call the custom DETR model from models.py
Instead of predicting bounding boxes, we will predict poles apex coordinates.
"""


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from models import build_model
import util.misc as utils
import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, choices=['resnet50', 'resnet101'])
    parser.add_argument('--position_embedding', default='sine', type=str, choices=['sine', 'learned'],
                        help="Type of positional embedding to use on top of the image features")
    
    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=20, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_apex', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    
    # Loss coefficients
    parser.add_argument('--apex_loss_coef', default=5, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    # dataset parameters
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters
    # TODO: later

    return parser



def main(args):
    """ Main function for model training."""
    device = torch.device(args.device)

    # Set seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)
    model.to(device)
    output_dir = Path(args.output_dir)
    print("Model successfully built.")
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # TODO: implement distributed training

    # Set up optimize and learning rate scheduler
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    print("Optimizer and learning rate scheduler successfully set up.")

    # Set up data loader
    assert args.data_path is not None, "You must provide a data path"
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)


    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, num_workers=args.num_workers)



    print("Dataset successfully built.")

    # Load checkpoint if asked to resume
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        # HERE YOU CAN REMOVE LAYERS FROM THE CHECKPOINT IF ARCHITECTURE IS DIFFERENT

        model.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheeduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # helps to resume training from pretrained model while keeping track of epochs
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        # TODO: implement evaluation
        pass

    print("Starting training...")
    start_time = time.time()

    # Training loop
    
    for epoch in range(args.start_epoch, args.epochs):
        # Train onche epoch
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, args.clip_max_norm)
        # update learning rate
        lr_scheduler.step()
        test_stats = evaluate(model, data_loader_val, device=device)
        # TODO: Log stats in a file

    # Save model checkpoint.pth
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, output_dir / 'checkpoint.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Custom DETR', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)