import numpy as np
import matplotlib.pyplot as plt
loss_data = np.load("Cyclical/train_val_loss.npz")

train_loss = loss_data['array1']
val_loss = loss_data['array2']
train_loss[train_loss > 0.1] = 0.1
# vgg19_accuracy = np.load("results/vgg19_accuracy.npz")

# vgg19_train_accs = vgg19_accuracy['array1']
# vgg19_val_accs = vgg19_accuracy['array2']

plt.figure(figsize=(10, 6))

# ResNet50 training and validation accuracy
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')

# VGG19 training and validation accuracy
# plt.plot(vgg19_train_accs, label='VGG19 Train Accuracy')
# plt.plot(vgg19_val_accs, label='VGG19 Validation Accuracy')

# Adding titles and labels
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("cyclical_loss_curve.png")