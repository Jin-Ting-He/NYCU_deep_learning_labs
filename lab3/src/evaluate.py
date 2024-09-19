import torch
from tqdm import tqdm
from utils import dice_score
def evaluate(model, val_dataloader, pbar_total):
    model.eval()
    pbar = tqdm(total=pbar_total, ncols=80)
    with torch.no_grad():
        total_dice = 0.0
        count = 0
        for sample in val_dataloader:
            image, mask = sample['image'].cuda(), sample['mask'].cuda()
            output = model(image)
            preds = torch.sigmoid(output) > 0.5  
            total_dice += dice_score(preds, mask)
            count += 1
            pbar.update(image.size(0))
        pbar.close()
    return total_dice / count