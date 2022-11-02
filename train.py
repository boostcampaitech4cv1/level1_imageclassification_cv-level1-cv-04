import argparse
import glob
import json
import multiprocessing
import os
import random
import re
import wandb
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from dataset import MaskBaseDataset
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
    
def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    
#     wandb.init(config={"batch_size": args.batch_size,
#                        "lr": args.lr,
#                        "epochs":args.epochs,
#                        "backborn":args.model,
#                        "UseCutmix":args.use_cutmix
                       
#     })
               

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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
#             wandb.log({'Valid loss': val_loss, 'Valid acc': val_acc})
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
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
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN','./data/train/images'))
#     '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    
    torch.cuda.empty_cache()
    
    train(data_dir, model_dir, args)



#     from model import *
#     device = "cuda"
# #     model = TinyVit_224(18)
#     model = Efficientb0(18)
#     model.load_state_dict(torch.load('model/exp_cutmix_1.0/best.pth'
#                                     ,map_location=device))
#     dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
#     dataset = dataset_module(
#         data_dir=data_dir,
#     )
#     model.cuda()
#     num_classes = dataset.num_classes  # 18

#     # -- augmentation
#     transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
#     transform = transform_module(
#         resize=args.resize,
#         mean=dataset.mean,
#         std=dataset.std,
#     )
    
#     val_transform_module = getattr(import_module("dataset"), 'ValAugmentation')  # validation을 위한 augmentation
#     val_transform = val_transform_module(
#         resize=args.resize,
#         mean=dataset.mean,
#         std=dataset.std,
#     )
    
#     dataset.set_transform(transform)

#     # -- data_loader
#     train_set, val_set = dataset.split_dataset()
#     val_set.dataset.transform = val_transform # valset의 trasnform을 따로 설정
#     print(val_set.dataset.transform)
    
#     val_loader = DataLoader(
#         val_set,
#         batch_size=args.valid_batch_size,
#         num_workers=multiprocessing.cpu_count() // 2,
#         shuffle=False,
#         pin_memory=True,
#         drop_last=False,
#     )
#     val_criterion = create_criterion(args.criterion)
#     with torch.no_grad():
#             print("Calculating validation results...")
#             model.eval()
#             val_loss_items = []
#             val_acc_items = []
#             figure = None
#             for val_batch in val_loader:
#                 inputs, labels = val_batch
#                 inputs = inputs['image'].to(device)
#                 labels = labels.to(device)

#                 outs = model(inputs)
#                 preds = torch.argmax(outs, dim=-1)

# #                 loss_item = criterion(outs, labels).item()
#                 loss_item = val_criterion(outs, labels).item()
    
#                 acc_item = (labels == preds).sum().item()
#                 val_loss_items.append(loss_item)
#                 val_acc_items.append(acc_item)

#                 if figure is None:
#                     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
#                     inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
#                     figure = grid_image(
#                         inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
#                     )

#             val_loss = np.sum(val_loss_items) / len(val_loader)
#             val_acc = np.sum(val_acc_items) / len(val_set)
#             print(
#                 f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
#             )
    
    
