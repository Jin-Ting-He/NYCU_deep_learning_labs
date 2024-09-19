import torch
from torch.utils.data import DataLoader
from dataloader import BufferflyMothLoader
from VGG19 import VGG19
import numpy as np
from tqdm import tqdm

def test_model(model, dataloader, loss_fn, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    running_corrects = 0
    total_samples = 0
    running_loss = 0.0

    with torch.no_grad():  # No need to track gradients for testing
        for images, labels in tqdm(dataloader, desc='Testing', ncols=80):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    accuracy = running_corrects.double() / total_samples

    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGG19(100).to(device)
    model_load_path = 'results/vgg19.pth'
    model.load_state_dict(torch.load(model_load_path))

    test_dataset = BufferflyMothLoader(root='dataset', mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()

    test_model(model, test_dataloader, loss_fn, device)