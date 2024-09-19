import numpy as np
import matplotlib.pyplot as plt
resnet50_accuracy = np.load("results/resnet50_accuracy.npz")

resnet50_train_accs = resnet50_accuracy['array1']
resnet50_val_accs = resnet50_accuracy['array2']

vgg19_accuracy = np.load("results/vgg19_accuracy.npz")

vgg19_train_accs = vgg19_accuracy['array1']
vgg19_val_accs = vgg19_accuracy['array2']

plt.figure(figsize=(10, 6))

# ResNet50 training and validation accuracy
plt.plot(resnet50_train_accs, label='ResNet50 Train Accuracy')
plt.plot(resnet50_val_accs, label='ResNet50 Validation Accuracy')

# VGG19 training and validation accuracy
plt.plot(vgg19_train_accs, label='VGG19 Train Accuracy')
plt.plot(vgg19_val_accs, label='VGG19 Validation Accuracy')

# Adding titles and labels
plt.title('Training and Validation Accuracy of ResNet50 and VGG19')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("results/accuracy_curve.png")