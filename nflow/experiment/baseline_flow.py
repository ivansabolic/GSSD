import torch
import time
import os

from .base import BaseExperiment
from .loss import elbo_bpd, elbo_nats
from .utils import get_args_table, clean_dict

from torch.utils.tensorboard import SummaryWriter
import wandb


class FlowExperiment(BaseExperiment):

    log_base = ''

    no_log_keys = ['project', 'name',
                   'log_tb', 'log_wandb',
                   'check_every', 'eval_every',
                   'device', 'parallel',
                   'pin_memory', 'num_workers']

    def __init__(self, args,
                 data_id, model_id, optim_id,
                 train_loader, eval_loader,
                 model, optimizer, scheduler_iter, scheduler_epoch):

        self.log_base = os.path.join(self.log_base, args.experiment_dir)

        if args.eval_every == -1:
            args.eval_every = args.epochs
        if args.check_every == -1:
            args.check_every = args.epochs
        if args.name == '':
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project == '':
            args.project = '_'.join([data_id, args.experiment_dir])

        model = model.to(args.device)

        log_path = os.path.join(self.log_base, args.name)
        super(FlowExperiment, self).__init__(model=model,
                                             optimizer=optimizer,
                                             scheduler_iter=scheduler_iter,
                                             scheduler_epoch=scheduler_epoch,
                                             log_path=log_path,
                                             eval_every=args.eval_every,
                                             check_every=args.check_every)

        self.create_folders()
        self.save_args(args)
        self.args = args

        self.data_id = data_id
        self.model_id = model_id
        self.optim_id = optim_id

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
        if args.log_tb:
            self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
            self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)
        if args.log_wandb != 'none':
            print("Logging to Weights & Biases")
            id = args.name + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            wandb.init(config=args_dict, project=args.project, id=id, dir=self.log_path)

    def log_fn(self, epoch, train_dict, eval_dict):
        if self.args.log_tb:
            for metric_name, metric_value in train_dict.items():
                self.writer.add_scalar('base/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    self.writer.add_scalar('eval/{}'.format(metric_name), metric_value, global_step=epoch+1)

        if self.args.log_wandb != 'none':
            for metric_name, metric_value in train_dict.items():
                wandb.log({'base/{}'.format(metric_name): metric_value}, step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    wandb.log({'eval/{}'.format(metric_name): metric_value}, step=epoch+1)

    def run(self):
        super(FlowExperiment, self).run(epochs=self.args.epochs)

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        t0 = time.time_ns()
        for x in self.train_loader:
            self.optimizer.zero_grad()
            loss = elbo_nats(self.model, x.to(self.args.device))
            loss.backward()
            if self.args.use_grad_norm > 0.:
                grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.args.use_grad_norm)
            else:
                grad_norm = torch.zeros(1)
            self.optimizer.step()
            if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Nats: {:.10f} Grad Norm: {:.3f}'.format(epoch + 1,
                                                                                                        self.args.epochs,
                                                                                                        loss_count, len(
                    self.train_loader.dataset), loss_sum / loss_count, grad_norm.item()), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        t1 = time.time_ns()
        print(f"Training time: {round((t1-t0)/(10**9), 2)} sec")
        return {'bpd': loss_sum/loss_count}

    def eval_fn(self, epoch):
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x in self.eval_loader:
                loss = elbo_bpd(self.model, x.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        return {'bpd': loss_sum/loss_count}


class OneClassFlowExperiment(FlowExperiment):
    def __init__(self, this_class_train_loader, this_class_eval_loader, other_classes_eval_loader, **kwargs):
        super(OneClassFlowExperiment, self).__init__(train_loader=this_class_train_loader,
                                                              eval_loader=this_class_eval_loader, **kwargs)
        self.this_class_train_loader = self.train_loader
        self.this_class_eval_loader = self.eval_loader
        self.other_classes_eval_loader = other_classes_eval_loader

    def eval_fn(self, epoch):
        return {
            'bpd_this_class': self._do_eval(epoch, self.this_class_eval_loader, 'This class')['bpd'],
            'bpd_other_classes': self._do_eval(epoch, self.other_classes_eval_loader, 'Other classes')['bpd'],
        }

    def _do_eval(self, epoch, eval_loader, metric):
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x in eval_loader:
                loss = elbo_nats(self.model, x.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Evaluating {}. Epoch: {}/{}, Datapoint: {}/{}, nats: {:.10f}'.format(metric, epoch + 1,
                                                                                            self.args.epochs,
                                                                                            loss_count,
                                                                                            len(eval_loader.dataset),
                                                                                            loss_sum / loss_count),
                      end='\r')
            print('')
        return {'bpd': loss_sum / loss_count}
