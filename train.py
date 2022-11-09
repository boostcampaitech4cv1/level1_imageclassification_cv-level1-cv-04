import argparse
import glob
import json
import multiprocessing
import os
import random
import re
import wandb
import math
import warnings
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold

from dataset import MaskBaseDataset, TestDataset
from cutmix import *
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr    

def age_converter(age):
    if age < 2: return 0
    if age < 5: return 1
    return 2
    
def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers, collator):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
#     val_set.dataset.transform = val_transform # valset의 trasnform을 따로 설정
    
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True,
        collate_fn = collator
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader
    
    
    
 # add mixup code
def mixup_data(x, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class Mixup_criterion:
    '''
    labels = (y_a, y_b, lam)
    '''
    def __init__(self, criterion):
        self.criterion = criterion
        
    def __call__(self, preds, labels):
        targets1, targets2, lam = labels
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)
    
def train_stratifiedkfold_tta(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))   

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)
    
    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    val_transform_module = getattr(import_module("dataset"), 'ValAugmentation')  # validation을 위한 augmentation
    val_transform = val_transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    dataset.set_transform(transform)
    
    labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]

    # -- data_loader
#     train_set, val_set = dataset.split_dataset()
    
#     val_set.dataset.transform = val_transform # valset의 trasnform을 따로 설정
    
    if args.use_cutmix:
        collator = CutMixCollator(args.cutmix_alpha)
    else:
        collator = torch.utils.data.dataloader.default_collate
        
    
    

    test_img_root = './back/back_test/images'
    test_info = pd.read_csv('./back/back_test/info.csv') 
    test_img_paths = [os.path.join(test_img_root, img_id) for img_id in test_info.ImageID]
    
    test_dataset = TestDataset(test_img_paths, args.resize)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    
    oof_pred = None
    patience = 8
    counter = 0
    
    
    for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
        
        num_workers=multiprocessing.cpu_count() // 2
        
        train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, args.batch_size, num_workers, collator)

        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(num_classes=num_classes).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
    #     criterion = create_criterion(args.criterion)  # default: cross_entropy
        if args.use_cutmix:
            train_criterion = CutMixCriterion(args)
        elif args.use_mixup:
            train_criterion = Mixup_criterion(create_criterion(args.criterion))
        else:
            train_criterion = create_criterion(args.criterion)  # default: cross_entropy
        val_criterion = create_criterion(args.criterion)

        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
#         scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5) 
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.001,  T_up=3, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs['image'].to(device)
    #             labels = labels.to(device) # torchvision은 output이 list, albermentation은 output이 dict이므로

                if args.use_cutmix:  # cutmix 추가하면서
                    targets1, targets2, lam = labels
                    labels = (targets1.to(device), targets2.to(device), lam)
                elif args.use_mixup:
                    inputs, targets1, targets2, lam = mixup_data(inputs, labels, args.alpha)
                    inputs, targets1, targets2 = map(Variable, (inputs, targets1, targets2))
                    labels = (targets1.to(device), targets2.to(device), lam)
                else:
                    labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)

                preds = torch.argmax(outs, dim=-1)
    #             loss = criterion(outs, labels)

                loss = train_criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
    #             matches += (preds == labels).sum().item()

                if args.use_cutmix:
                    targets1, targets2, lam = labels
                    cor1 = (preds == targets1).sum().item()
                    cor2 = (preds == targets2).sum().item()
                    matches += (lam * cor1 + (1 - lam) * cor2)
                elif args.use_mixup:
                    targets1, targets2, lam = labels
                    matches += (lam * preds.eq(targets1.data).cpu().sum().float()
                        + (1 - lam) * preds.eq(targets2.data).cpu().sum().float())
                else:
                    matches += (preds == labels).sum().item()


                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
    #                 wandb.log({'Train loss': train_loss, 'Train acc': train_acc})

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs['image'].to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

    #                 loss_item = criterion(outs, labels).item()
                    loss_item = val_criterion(outs, labels).item()

                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / (len(val_loader)*args.batch_size)
                best_val_acc = max(best_val_acc, val_acc)
                if val_loss < best_val_loss:
                    print(f"New best model for val loss : {val_loss:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_model = model
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                
                
                
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                
                if counter > patience:
                    print("Early Stopping...")
                    break
    #             wandb.log({'Valid loss': val_loss, 'Valid acc': val_acc})
                print()
        
        all_predictions = []
        with torch.no_grad():
            for images in test_loader:
                images = images['image'].to(device)

                # Test Time Augmentation
                pred = best_model(images) / 2 # 원본 이미지를 예측하고
                pred += best_model(torch.flip(images, dims=(-1,))) / 2 # horizontal_flip으로 뒤집어 예측합니다. 
                all_predictions.extend(pred.cpu().numpy())

            fold_pred = np.array(all_predictions)

        # 확률 값으로 앙상블을 진행하기 때문에 'k'개로 나누어줍니다.
        if oof_pred is None:
            oof_pred = fold_pred / n_splits
        else:
            oof_pred += fold_pred / n_splits
        
        
        oof_pred_list = []
        if i == 4:
            for images in test_loader:
                loaded_model = torch.load_state_dict(torch.load())
            for row in range(len(oof_pred)):
                oof_pred_list.append(str(oof_pred[row]))
            
            test_info['logit'] = oof_pred_list
            save_logit_path = './output/output_kfold_logit.csv'
            test_info.to_csv(save_logit_path, index=False)
            
            oof_pred = torch.from_numpy(oof_pred)
            oof_pred_idx = oof_pred.argmax(dim=-1)
            test_info['ans'] = oof_pred_idx
            save_answer_path = './output/output_kfold_answer.csv'
            test_info.to_csv(save_answer_path, index=False)
            print(f"Inference Done! Inference result saved at {save_path}")
            
            
def train_cutmix(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)
    
    # -- dataset
    dataset_module = getattr(import_module("dataset"), 'MaskSplitByProfileDataset')  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    val_transform_module = getattr(import_module("dataset"), 'ValAugmentation')  # validation을 위한 augmentation
    val_transform = val_transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    dataset.set_transform(transform)


#     -- data_loader
    train_set, val_set = dataset.split_dataset()
    
    val_set.dataset.transform = val_transform # valset의 trasnform을 따로 설정
    
    if args.use_cutmix:
        collator = CutMixCollator(args.cutmix_alpha)
    else:
        collator = torch.utils.data.dataloader.default_collate
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
        collate_fn = collator
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    
    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
#     criterion = create_criterion(args.criterion)  # default: cross_entropy
    if args.use_cutmix:
        train_criterion = CutMixCriterion(args)
    elif args.use_mixup:
        train_criterion = Mixup_criterion(create_criterion(args.criterion))
    else:
        train_criterion = create_criterion(args.criterion)  # default: cross_entropy
    val_criterion = create_criterion(args.criterion)

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5) 

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs['image'].to(device)
#             labels = labels.to(device) # torchvision은 output이 list, albermentation은 output이 dict이므로


            if args.use_cutmix:  # cutmix 추가하면서
                targets1, targets2, lam = labels
                labels = (targets1.to(device), targets2.to(device), lam)
            elif args.use_mixup:
                inputs, targets1, targets2, lam = mixup_data(inputs, labels, args.alpha)
                inputs, targets1, targets2 = map(Variable, (inputs, targets1, targets2))
                labels = (targets1.to(device), targets2.to(device), lam)
            else:
                labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)

            preds = torch.argmax(outs, dim=-1)
#             loss = criterion(outs, labels)

            loss = train_criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
#             matches += (preds == labels).sum().item()

            if args.use_cutmix:
                targets1, targets2, lam = labels
                cor1 = (preds == targets1).sum().item()
                cor2 = (preds == targets2).sum().item()
                matches += (lam * cor1 + (1 - lam) * cor2)
            elif args.use_mixup:
                targets1, targets2, lam = labels
                matches += (lam * preds.eq(targets1.data).cpu().sum().float()
                    + (1 - lam) * preds.eq(targets2.data).cpu().sum().float())
            else:
                matches += (preds == labels).sum().item()


            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs['image'].to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

#                 loss_item = criterion(outs, labels).item()
                loss_item = val_criterion(outs, labels).item()

                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_acc = max(best_val_acc, val_acc)
            if val_loss < best_val_loss:
                print(f"New best model for val loss : {val_loss:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_loss = val_loss
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)

            print()


def train_multi_label(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), 'MultiMaskSplitByProfileDataset')  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    val_transform_module = getattr(import_module("dataset"), 'ValAugmentation')  # validation을 위한 augmentation
    val_transform = val_transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    dataset.set_transform(transform)
    
    # -- data_loader
    train_set, val_set = dataset.split_dataset()


    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            
            inputs, (mask_labels, gender_labels, age_labels) = train_batch
            inputs = inputs['image'].to(device)
            
            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)
            
            optimizer.zero_grad()
            
            outs = model(inputs)
            
            (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 6], dim=1)
            
            mask_loss = criterion(mask_outs, mask_labels, 3)
            gender_loss = criterion(gender_outs, gender_labels, 2)
            age_loss = criterion(age_outs, age_labels, 6)
            
            mask_preds = torch.argmax(mask_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            
            age_probs = torch.nn.functional.softmax(age_outs)
            age_probs = torch.transpose(age_probs,0,1)
            age_probs1 = torch.transpose(torch.unsqueeze(torch.sum(age_probs[:2], dim=0), 0),0,1)
            age_probs2 = torch.transpose(torch.unsqueeze(torch.sum(age_probs[2:5], dim=0), 0),0,1)
            age_probs3 = torch.transpose(age_probs[5:],0,1)
            age_add_probs = torch.cat((age_probs1, age_probs2, age_probs3), -1)
            age_preds = torch.argmax(age_add_probs, dim=-1)
            
            #age_preds = torch.argmax(age_outs, dim=-1)
            
            #age_preds = torch.tensor([age_converter(age) for age in age_preds])
            age_labels = torch.tensor([age_converter(age) for age in age_labels])
            age_preds = age_preds.to(device)
            age_labels = age_labels.to(device)
            
            
            preds = age_preds + 3*gender_preds + 6*mask_preds
            labels = age_labels + 3*gender_labels + 6*mask_labels
            
            loss = mask_loss + gender_loss + age_loss
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            mask_loss_items = []
            age_loss_items = []
            gender_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, (mask_labels, gender_labels, age_labels) = val_batch
                # 3 6 2
                inputs = inputs['image'].to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)

                outs = model(inputs)
                
                (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 6], dim=1)
                
                mask_loss = criterion(mask_outs, mask_labels, 3)
                gender_loss = criterion(gender_outs, gender_labels, 2)
                age_loss = criterion(age_outs, age_labels, 6)
                
                mask_preds = torch.argmax(mask_outs, dim=-1)
                gender_preds = torch.argmax(gender_outs, dim=-1)
                
                age_probs = torch.nn.functional.softmax(age_outs)
                age_probs = torch.transpose(age_probs,0,1)
                age_probs1 = torch.transpose(torch.unsqueeze(torch.sum(age_probs[:2], dim=0), 0),0,1)
                age_probs2 = torch.transpose(torch.unsqueeze(torch.sum(age_probs[2:5], dim=0), 0),0,1)
                age_probs3 = torch.transpose(age_probs[5:],0,1)
                age_add_probs = torch.cat((age_probs1/2, age_probs2/3, age_probs3), -1)
                age_preds = torch.argmax(age_add_probs, dim=-1)
                
                age_labels = torch.tensor([age_converter(age) for age in age_labels])
                age_preds = age_preds.to(device)
                age_labels = age_labels.to(device)
                
                preds = age_preds + 3*gender_preds + 6*mask_preds
                labels = age_labels + 3*gender_labels + 6*mask_labels
                loss = mask_loss + gender_loss + age_loss

                loss_item = loss.item()
                mask_loss_item = mask_loss.item()
                gender_loss_item = gender_loss.item()
                age_loss_item = age_loss.item()
                
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                mask_loss_items.append(mask_loss_item)
                gender_loss_items.append(gender_loss_item)
                age_loss_items.append(age_loss_item)
                
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            mask_val_loss = np.sum(mask_loss_items) / len(val_loader)
            gender_val_loss = np.sum(gender_loss_items) / len(val_loader)
            age_val_loss = np.sum(age_loss_items) / len(val_loader)
            
            print(f"mask_loss: {mask_val_loss:4.4}")
            print(f"gender_loss: {gender_val_loss:4.4}")
            print(f"age_loss: {age_val_loss:4.4}")
            
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_acc = max(best_val_acc, val_acc)
            if val_loss < best_val_loss:
                print(f"New best model for val loss : {val_loss:4.4}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_loss = val_loss
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.4} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.4}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()            
        
def train_cutmix_multilabel(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), 'MaskSplitByProfileDataset36')  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = 36  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    val_transform_module = getattr(import_module("dataset"), 'ValAugmentation')  # validation을 위한 augmentation
    val_transform = val_transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()
    
    val_set.dataset.transform = val_transform # valset의 trasnform을 따로 설정
    
    if args.use_cutmix:
        collator = CutMixCollator(args.cutmix_alpha)
    else:
        collator = torch.utils.data.dataloader.default_collate
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
        collate_fn = collator
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=11
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
#     criterion = create_criterion(args.criterion)  # default: cross_entropy
    if args.use_cutmix:
        train_criterion = CutMixCriterion(args)
    elif args.use_mixup:
        train_criterion = Mixup_criterion(create_criterion(args.criterion))
    else:
        train_criterion = create_criterion(args.criterion)  # default: cross_entropy
    val_criterion = create_criterion(args.criterion)
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5) 

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs['image'].to(device)
#             labels = labels.to(device) # torchvision은 output이 list, albermentation은 output이 dict이므로
            
            if args.use_cutmix:  # cutmix 추가하면서
                targets1, targets2, lam = labels
                labels = (targets1.to(device), targets2.to(device), lam)
            elif args.use_mixup:
                inputs, targets1, targets2, lam = mixup_data(inputs, labels, args.alpha)
                inputs, targets1, targets2 = map(Variable, (inputs, targets1, targets2))
                labels = (targets1.to(device), targets2.to(device), lam)
            else:
                labels = labels.to(device)

            optimizer.zero_grad()
            
            outs = model(inputs)
            
            r_outs = torch.permute(outs, (1,0))
            mask_outs = torch.permute(r_outs[:3], (1,0))
            gender_outs = torch.permute(r_outs[3:5], (1,0))
            age_outs = torch.permute(r_outs[5:], (1,0))
            
            mask_targets1 = (targets1 // 12) %3
            mask_targets2 = (targets2 // 12) %3
            gender_targets1 = (targets1 // 6) %2
            gender_targets2 = (targets2 // 6) %2
            age_targets1 = targets1 % 6
            age_targets2 = targets2 % 6
            
            mask_targets1 = mask_targets1.to(device)
            mask_targets2 = mask_targets2.to(device)
            gender_targets1 = gender_targets1.to(device)
            gender_targets2 = gender_targets2.to(device)
            age_targets1 = age_targets1.to(device)
            age_targets2 = age_targets2.to(device)
            
            mask_labels = (mask_targets1, mask_targets2, lam)
            gender_labels = (gender_targets1, gender_targets2, lam)
            age_labels = (age_targets1, age_targets2, lam)
            
            mask_preds = torch.argmax(mask_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            age_preds = torch.argmax(age_outs, dim=-1)
            
            preds = mask_preds * 12 + gender_preds * 6 + age_preds
            
#             loss = criterion(outs, labels)
            
            mask_loss = train_criterion(mask_outs, mask_labels, 3)
            gender_loss = train_criterion(gender_outs, gender_labels, 2)
            age_loss = train_criterion(age_outs, age_labels, 6)
            
            loss = mask_loss + gender_loss + age_loss
            loss.requires_grad_(True)
            
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
#             matches += (preds == labels).sum().item()

            if args.use_cutmix:
                targets1, targets2, lam = labels
                cor1 = (preds == targets1).sum().item()
                cor2 = (preds == targets2).sum().item()
                
                matches += (lam * cor1 + (1 - lam) * cor2)

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs['image'].to(device)
                labels = labels.to(device)

                outs = model(inputs)
                
                r_outs = torch.permute(outs, (1,0))
                mask_outs = torch.permute(torch.tensor(r_outs[:3]), (1,0))
                gender_outs = torch.permute(torch.tensor(r_outs[3:5]), (1,0))
                age_outs = torch.permute(torch.tensor(r_outs[5:]), (1,0))
                
                mask_preds = torch.argmax(mask_outs, dim=-1)
                gender_preds = torch.argmax(gender_outs, dim=-1)
                age_preds = torch.argmax(age_outs, dim=-1)
                
                preds = mask_preds * 12 + gender_preds * 6 + age_preds
                
                #preds = torch.argmax(outs, dim=-1)

#                 loss_item = criterion(outs, labels).item()

                mask_labels = (labels // 12) %3
                gender_labels = (labels // 6) %2
                age_labels = labels % 6
                
                mask_loss = val_criterion(mask_outs, mask_labels, 3)
                gender_loss = val_criterion(gender_outs, gender_labels, 2)
                age_loss = val_criterion(age_outs, age_labels, 6)

                loss = mask_loss + gender_loss + age_loss
                loss_item = loss.item()

                #loss_item = val_criterion(outs, labels, 36).item()
    
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_acc = max(best_val_acc, val_acc)
            if val_loss < best_val_loss:
                print(f"New best model for val loss : {val_loss:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_loss = val_loss
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)

            print()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=128, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='TinyVit_224', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='f1', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--use_cutmix', action='store_true')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Data Cutmix')
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--alpha', default=0.2, type=float, help='mixup interpolation coefficient (default: 1)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN','./back/back_train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train_mode', type=str, default='train_cutmix') # train_cutmix, train_stratifiedkfold_tta, train_multi_label,train_cutmix_multilabel

    args = parser.parse_args()
    print(args)

    
    warnings.filterwarnings(action='ignore')
    data_dir = args.data_dir
    model_dir = args.model_dir
    train_mode = args.train_mode
    torch.cuda.empty_cache()
    
    if train_mode=='train_cutmix':
        train_cutmix(data_dir, model_dir, args)
    elif train_mode=='train_stratifiedkfold_tta':
        train_stratifiedkfold_tta(data_dir, model_dir, args)
    elif train_mode=='train_multi_label':
        train_multi_label(data_dir, model_dir, args)
    else:
        train_cutmix_multilabel(data_dir, model_dir, args)
