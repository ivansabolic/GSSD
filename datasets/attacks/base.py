import glob
import os
import os.path as osp
import time
from copy import deepcopy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10, GTSRB
from tqdm import tqdm
import wandb

from ..utils import Log

support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10,
    GTSRB,
)


def check(dataset):
    return isinstance(dataset, support_list)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




class Base(object):
    """Base class for backdoor training and testing.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        schedule (dict): Training or testing global schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self, train_dataset, test_dataset, model, loss, schedule=None, seed=0, deterministic=False):
        assert isinstance(train_dataset,
                          support_list), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        self.train_dataset = train_dataset

        assert isinstance(test_dataset,
                          support_list), 'test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list.'
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.global_schedule = deepcopy(schedule)
        self.current_schedule = None
        self._set_seed(seed, deterministic)

        self.normalizer = None

    def _set_seed(self, seed, deterministic):
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
            # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_model(self):
        return self.model

    def get_poisoned_dataset(self):
        return self.poisoned_train_dataset, self.poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule['schedule']:
            self.current_schedule['lr'] *= self.current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.current_schedule['lr']

    def train(self, schedule=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if 'pretrain' in self.current_schedule:
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain']), strict=False)

        device = self.current_schedule['device']

        if self.current_schedule['benign_training'] is True:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
        elif self.current_schedule['benign_training'] is False:
            train_loader = DataLoader(
                self.poisoned_train_dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
        else:
            raise AttributeError("self.current_schedule['benign_training'] should be True or False.")

        self.model = self.model.to(device)
        self.model.train()

        encoder_params = []
        linear_params = []

        # Iterate over named parameters
        for name, param in self.model.named_parameters():
            if 'linear' in name:
                linear_params.append(param)
            else:
                encoder_params.append(param)

        if 'backbone_lr' in self.current_schedule and self.current_schedule['backbone_lr'] and self.current_schedule[
            'backbone_lr'] != self.current_schedule['lr']:
            print('Using different learning rates for backbone and linear layers.')
            optimizer = torch.optim.SGD([
                {'params': encoder_params, 'lr': self.current_schedule['backbone_lr']},
                {'params': linear_params, 'lr': self.current_schedule['lr']}
            ], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'],
                                        momentum=self.current_schedule['momentum'],
                                        weight_decay=self.current_schedule['weight_decay'])

        defense = False
        if self.current_schedule['benign_training'] is True:
            exp_info = 'benign_training'
        elif not self.current_schedule['defense']:
            exp_info = 'poisoned_training'
        else:
            if 'ransac' in self.current_schedule['indices']:
                exp_info = 'defense_ransac'
            else:
                defense = True
                exp_info = 'defense'
                if 'projections' in self.current_schedule['indices'].split('/')[-2]:
                    exp_info += '_projections'
                if self.current_schedule['indices'].split('/')[-2].endswith('retrain'):
                    exp_info += '_retrain'
                if 'load_pretrained' in self.current_schedule and self.current_schedule['load_pretrained'] is True:
                    exp_info += '_pretrained'

        if 'add_exp_info' in self.current_schedule and self.current_schedule['add_exp_info'] != '':
            exp_info += '_' + self.current_schedule['add_exp_info']

        exp_name = exp_info
        if defense is True:
            ind_name = self.current_schedule['indices'][
                       self.current_schedule['indices'].index("indices_") + len("indices_"):-4]
            exp_info += '_' + ind_name
        self.current_schedule['exp_info'] = exp_info

        work_dir = osp.join(self.current_schedule['experiment_dir'], exp_info)

        if 'resume' in self.current_schedule and self.current_schedule['resume'] is True:
            # Look for checkpoint in the base defense directory (without add_exp_info)
            resume_dir = work_dir
            if 'add_exp_info' in self.current_schedule and self.current_schedule['add_exp_info'] != '':
                base_exp_info = exp_info.replace('_' + self.current_schedule['add_exp_info'], '', 1)
                base_dir = osp.join(self.current_schedule['experiment_dir'], base_exp_info)
                if osp.exists(base_dir) and not osp.exists(resume_dir):
                    resume_dir = base_dir
            # glob all .pth files
            ckpt_files = sorted(glob.glob(osp.join(resume_dir, '*.pth')))
            assert len(ckpt_files) > 0, f"No checkpoint files found in {resume_dir}"

            # load the last checkpoint
            self.model.load_state_dict(torch.load(ckpt_files[0]), strict=False)
            print(f"Resume training from {ckpt_files[0]}")
            work_dir = osp.join(work_dir, 'resumed')
            exp_name = 'resumed_' + exp_name

        if os.path.exists(work_dir):
            import warnings
            warnings.warn(f"Log folder {work_dir} already exists, overwriting.")

        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        exp_name = f"{self.current_schedule['experiment_dir'].split('/')[-1]}_{exp_name}"
        if self.current_schedule['wandb'] is True:
            wandb.init(
                project="BackdoorBox",
                id=exp_name + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
                name=exp_name,
                config=self.current_schedule,
                dir=work_dir,
            )
        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        iteration = 0
        last_time = time.time()

        len_dataset = len(self.train_dataset) if self.current_schedule['benign_training'] is True else len(
            self.poisoned_train_dataset)
        msg = f"Total train samples: {len_dataset}\n" \
              f"Total test samples: {len(self.test_dataset)}\n" \
              f"Batch size: {self.current_schedule['batch_size']}\n" \
              f"iteration every epoch: {len_dataset // self.current_schedule['batch_size']}\n" \
              f"Initial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        self._test_and_log(self.test_dataset, device, 'benign', 0, log)
        self._test_and_log(self.asr_dataset, device, 'ASR', 0, log)
        for i in range(self.current_schedule['epochs']):
            self.adjust_learning_rate(optimizer, i)
            for batch_id, batch in enumerate(tqdm(train_loader)):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = self.model(batch_img)
                loss = self.loss(predict_digits, batch_label)
                loss.backward()
                optimizer.step()

                iteration += 1

                # if iteration % self.current_schedule['log_iteration_interval'] == 0:
            msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ",
                                time.localtime()) + f"Epoch:{i + 1}/{self.current_schedule['epochs']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time() - last_time}\n"
            last_time = time.time()
            log(msg)
            if self.current_schedule['wandb'] is True:
                wandb.log({'loss': float(loss), 'lr': self.current_schedule['lr']}, step=i + 1)

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                # test result on benign test dataset
                self._test_and_log(self.test_dataset, device, 'benign', i, log)
                self._test_and_log(self.poisoned_test_dataset, device, 'poisoned', i, log)
                self._test_and_log(self.asr_dataset, device, 'ASR', i, log)
                self._test_and_log(self.target_class_dataset, device, 'target_class', i, log)

                self.model = self.model.to(device)
                self.model.train()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0 or i == self.current_schedule['epochs'] - 1:
                self.model.eval()
                self.model = self.model.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(i + 1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model = self.model.to(device)
                self.model.train()

    def _test_and_log(self, dataset, device, dataset_name, i, log):
        start_time = time.time()
        predict_digits, labels = self._test(dataset, device, self.current_schedule['batch_size'],
                                            self.current_schedule['num_workers'])
        total_num = labels.size(0)
        prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
        top1_correct = int(round(prec1.item() / 100.0 * total_num))
        top5_correct = int(round(prec5.item() / 100.0 * total_num))
        msg = f"==========Test result on {dataset_name} test dataset==========\n" + \
              time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
              f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct / total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct / total_num}, time: {time.time() - start_time}\n"
        log(msg)
        if self.current_schedule['wandb'] is True:
            wandb.log({f'{dataset_name}_top1_acc': top1_correct / total_num},
                      step=i + 1)

    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None):
        if model is None:
            model = self.model
        else:
            model = model

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            model = model.to(device)
            model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                if self.normalizer:
                    batch_img = self.normalizer(batch_img)
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels

    def test(self, schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in self.current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert self.current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(
                f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

            if self.current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(self.current_schedule['GPU_num']))
                model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_dir'], 'retraining')
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log_test.txt'))

        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            predict_digits, labels = self._test(test_dataset, device, self.current_schedule['batch_size'],
                                                self.current_schedule['num_workers'], model)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct / total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct / total_num}, time: {time.time() - last_time}\n"
            log(msg)

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            predict_digits, labels = self._test(poisoned_test_dataset, device, self.current_schedule['batch_size'],
                                                self.current_schedule['num_workers'], model)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct / total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct / total_num}, time: {time.time() - last_time}\n"
            log(msg)
