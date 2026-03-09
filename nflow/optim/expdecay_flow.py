import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from .schedulers import LinearWarmupScheduler

optim_choices = {'sgd', 'adam', 'adamax', 'adamw'}




def get_optim_id(args):
    return 'expdecay'
    # return f'{args.optimizer}_lr{args.lr}_mom{args.momentum}_mom2{args.momentum_sqr}_gamma{args.gamma}'


def get_optim(args, model):
    assert args.optimizer in optim_choices

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr))
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr))
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr), weight_decay=0.01)

    if args.warmup != -1:
        scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=args.warmup)
    else :
        scheduler_iter = None

    scheduler_epoch = ExponentialLR(optimizer, gamma=args.gamma)


    return optimizer, scheduler_iter, scheduler_epoch