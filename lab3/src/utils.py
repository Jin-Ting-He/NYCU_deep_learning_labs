import matplotlib.pyplot as plt
import numpy as np
import torch

def count_same_pixels(tensor1, tensor2):
    """
    计算两个二进制Tensor第一个元素中相同像素的数量。

    参数:
    - tensor1: 第一个输入Tensor，形状为[N, C, H, W]，其中N是批次大小。
    - tensor2: 第二个输入Tensor，形状应与tensor1相同。

    返回:
    - same_pixel_count: 第一个元素中相同像素的数量。
    """
    if tensor1.size() != tensor2.size():
        raise ValueError("两个Tensor的形状必须相同。")

    # 从每个Tensor中取出第一个元素
    first_tensor1 = tensor1[0]
    first_tensor2 = tensor2[0]

    # 比较两个Tensor中的像素值是否相同
    same_pixels = torch.eq(first_tensor1, first_tensor2)

    # 计算相同像素的数量
    same_pixel_count = torch.sum(same_pixels).item()
    print(same_pixel_count)
    # return same_pixel_count

def dice_score(preds, labels):
    smooth = 1e-6  # 防止除以零
    preds = preds.contiguous().float()  # 确保是浮点类型
    labels = labels.contiguous().float()
    intersection = (preds * labels).sum(dim=[2, 3])
    dice = (2. * intersection + smooth) / (preds.sum(dim=[2, 3]) + labels.sum(dim=[2, 3]) + smooth)

    return dice.mean()

def plot_dice_score_curve(train_dice_scores, valid_dice_scores, epochs, model_name):
    plt.plot(range(1, epochs + 1), train_dice_scores, label='Training Dice Score')
    plt.plot(range(1, epochs + 1), valid_dice_scores, label='Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title(model_name + ' Dice Score over Epochs')
    plt.legend()
    plt.savefig(model_name + ' Dice Score over Epochs')
    plt.close()

if __name__ == "__main__":
    tensor1 = torch.tensor([[[[0, 0], [1, 0]]]], dtype=torch.float32)  # 形状为[1, 1, 2, 2]
    tensor2 = torch.tensor([[[[1, 1], [0, 1]]]], dtype=torch.float32)  # 形状为[1, 1, 2, 2]

    # 调用函数并打印相同像素的数量
    count = count_same_pixels(tensor1, tensor2)
    print(f"Number of same pixels in the first elements: {count}")

    d_score = dice_score(tensor1, tensor2)
    print(f"Dice score: {d_score}")