import argparse
import multiprocessing
import os
import warnings 

from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference_cutmix(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images['image'].to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


@torch.no_grad()
def inference_multi_label(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    num_classes = 11
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images['image'].to(device)
            #pred = model(images)
            
            outs = model(images)
            (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 6], dim=1)
            
            mask_preds = torch.argmax(mask_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            
            
            age_probs = torch.nn.functional.softmax(age_outs)
            age_probs = torch.transpose(age_probs,0,1)
            age_probs1 = torch.transpose(torch.unsqueeze(torch.sum(age_probs[:2], dim=0), 0),0,1)
            age_probs2 = torch.transpose(torch.unsqueeze(torch.sum(age_probs[2:5], dim=0), 0),0,1)
            age_probs3 = torch.transpose(age_probs[5:],0,1)
            age_add_probs = torch.cat((age_probs1, age_probs2, age_probs3), -1)
            age_preds = torch.argmax(age_add_probs, dim=-1)
            
            
            age_preds = age_preds.to(device)
            
            pred = age_preds + 3*gender_preds + 6*mask_preds
            
            #pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")

def age_converter(age):
    if age<2: return 0
    if age<5: return 1
    return 2

@torch.no_grad()
def inference_cutmix_multi_label(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    num_classes = 11
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images['image'].to(device)
            
            outs = model(images)
            
            r_outs = torch.permute(outs, (1,0))
            mask_outs = torch.permute(torch.tensor(r_outs[:3]), (1,0))
            gender_outs = torch.permute(torch.tensor(r_outs[3:5]), (1,0))
            age_outs = torch.permute(torch.tensor(r_outs[5:]), (1,0))
            
            mask_preds = torch.argmax(mask_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            age_preds = torch.argmax(age_outs, dim=-1)
            age_preds = torch.tensor([age_converter(age) for age in age_preds])
            age_preds = age_preds.to(device)
            
            pred = mask_preds * 6 + gender_preds * 3 + age_preds
            
            #pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for validing (default: 128)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (224, 224))')
    parser.add_argument('--model', type=str, default='TinyVit_224', help='model type (default: TinyVit_224)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', './data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--inference_mode', type=str, default='train_cutmix') # inference_cutmix, inference_multi_label,inference_cutmix_multi_label
    args = parser.parse_args()

    warnings.filterwarnings(action='ignore')
    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    inference_mode = args.inference_mode
    os.makedirs(output_dir, exist_ok=True)

    
    # stratifiedkfold_tta의 경우 train이 끝나면 이어서 inference
    if inference_mode=='inference_cutmix':
        inference_cutmix(data_dir, model_dir, output_dir, args)
    elif inference_mode=='inference_multi_label':
        inference_multi_label(data_dir, model_dir, output_dir, args)
    else:
        inference_cutmix_multi_label(data_dir, model_dir, output_dir, args)
