import numpy as np 
#Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import pdb
#CV
import cv2

import sys
sys.path.append('/nfs/hpc/share/karkisa/Thesis (Action_Seg)/Research paper implementations/Wheat_Detection/detr')

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

#Albumenatations
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2

import warnings
warnings.filterwarnings('ignore')

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

from Pipeline import display_,WheatDataset, get_valid_transforms
from Model import DETRModel
from Logic import classifier, collate_fn

def view_sample(df_valid,marking,model,device):
    '''
    Code taken from Peter's Kernel 
    https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train
    '''
    valid_dataset = WheatDataset(image_ids=df_valid.index.values,
                                 dataframe=marking,
                                 transforms=get_valid_transforms()
                                )
     
    valid_data_loader = DataLoader(
                                    valid_dataset,
                                    batch_size=32,
                                    shuffle=False,
                                   num_workers=4,
                                   collate_fn=collate_fn)
    
    images, targets = next(iter(valid_data_loader))
    _,h,w = images[0].shape # for de normalizing images
    
    images = list(img.to(device) for img in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    boxes = targets[0]['boxes'].cpu().numpy()
    boxes = [np.array(box).astype(np.int32) for box in A.augmentations.bbox_utils.denormalize_bboxes(boxes,h,w)]
    sample = images[0].permute(1,2,0).cpu().numpy()
    
    model.eval()
    model.to(device)
    cpu_device = torch.device("cpu")
    
    with torch.no_grad():
        outputs = model(images)
        
    outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2]+box[0], box[3]+box[1]),
                  (220, 0, 0), 1)
        

    oboxes = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
    oboxes = [np.array(box).astype(np.int32) for box in A.augmentations.bbox_utils.denormalize_bboxes(oboxes,h,w)]
    prob   = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:,0]
    
    for box,p in zip(oboxes,prob):
        
        # if p >0.5:
        color = (0,0,220) #if p>0.5 else (0,0,0)
        cv2.rectangle(sample,
                (box[0], box[1]),
                (box[2]+box[0], box[3]+box[1]),
                color, 1)
    
    ax.set_axis_off()
    ax.imshow(sample)
    plt.savefig("pred_1.png")

    # plt.show()

def main(fold=1):
    train_df_path='/nfs/hpc/share/karkisa/Thesis (Action_Seg)/Research paper implementations/Wheat_Detection/train.csv'
    fold_df,marking=display_(train_df_path)
    df_valid=fold_df[fold_df['fold']==fold]
    model=DETRModel(num_classes=2,num_queries=100)
    path='/nfs/hpc/share/karkisa/Thesis (Action_Seg)/Research paper implementations/Wheat_Detection/lightning_logs/version_7/checkpoints/epoch=34-step=1504.ckpt'
    ckptt=torch.load(path,
    # , map_location=torch.device('cpu')
                    )
    model=nn.DataParallel(model)
    model.load_state_dict(ckptt['state_dict'],strict=False)
    view_sample(df_valid,marking,model,'cuda')
    
main(1)