import argparse
import time

import torch

import datasets
from utils import yaml_config_hook
from nflow.utils import set_seeds
from nflow.experiment.baseline_flow import FlowExperiment, OneClassFlowExperiment
from nflow.dataset.features import get_data_id, get_data_oneclass, get_data_allclasses
from nflow.model.mini_flow import get_model, get_model_id
from nflow.optim import get_optim, get_optim_id


def main(args):
    args.num_classes = datasets.dataset_num_clases[args.dataset]

    if args.oneclass == -1:
        train_loader, eval_loader, data_shape = get_data_allclasses(args, unsupervised=True)
    else:
        train_loader, eval_loader, data_shape = get_data_oneclass(args)

    data_id = get_data_id(args)
    model = get_model(args, data_shape=data_shape)
    model_id = get_model_id(args)
    optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
    optim_id = get_optim_id(args)

    print(args)
    torch.backends.cudnn.benchmark = True
    start_time = time.time()

    if args.oneclass == -1:
        exp = FlowExperiment(args=args,
                             data_id=data_id,
                             model_id=model_id,
                             optim_id=optim_id,
                             train_loader=train_loader[0],
                             eval_loader=eval_loader[0],
                             model=model,
                             optimizer=optimizer,
                             scheduler_iter=scheduler_iter,
                             scheduler_epoch=scheduler_epoch)
    else:
        exp = OneClassFlowExperiment(args=args,
                                     data_id=data_id,
                                     model_id=model_id,
                                     optim_id=optim_id,
                                     this_class_train_loader=train_loader[0],
                                     this_class_eval_loader=eval_loader[0],
                                     other_classes_eval_loader=eval_loader[1],
                                     model=model,
                                     optimizer=optimizer,
                                     scheduler_iter=scheduler_iter,
                                     scheduler_epoch=scheduler_epoch)

    exp.run()
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/nflow_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    set_seeds(args.seed)
    main(args)
