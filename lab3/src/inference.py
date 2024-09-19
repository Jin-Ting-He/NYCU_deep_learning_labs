import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.unet import UNet
from models.resnet34_unet import UNetPlusResNet34
from oxford_pet import load_dataset
from evaluate import evaluate
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default='dataset/oxford-iiit-pet', help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    print("########### UNet Inference ###########")
    model = UNet(3,1).cuda()
    model_load_path = 'saved_models/DL_Lab3_UNet_312551065_何勁廷.pth'
    model.load_state_dict(torch.load(model_load_path))
    test_dataset = load_dataset(data_path=args.data_path, mode='test') 
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=16, shuffle=False)
    dice_score = round(float(evaluate(model, test_dataloader, len(test_dataset))),4)
    print("UNet Dice Score: ", dice_score)

    print("########### ResNet34+UNet Inference ###########")
    model = UNetPlusResNet34(3,1).cuda()
    model_load_path = 'saved_models/DL_Lab3_ResNet34_UNet_312551065_何勁廷.pth'
    model.load_state_dict(torch.load(model_load_path))

    test_dataset = load_dataset(data_path=args.data_path, mode='test') 
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=16, shuffle=False)

    dice_score = round(float(evaluate(model, test_dataloader, len(test_dataset))), 4)

    print("ResNet34+UNet Dice Score: ", dice_score)

    