import numpy as np
import matplotlib.pyplot as plt
def generate_linear(n = 100):
    """
    Generate data points which are linearly separable
    :param n: number of points
    :return: inputs and labels
    """
    pts = np.random.uniform(0, 1, (n, 2))
    inputs, labels = [], []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        
        if pt[0] > pt[1]:
            labels.append(0)
        else: 
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    """
    Generate data points based on XOR situation
    :param n: number of points
    :return: inputs and labels
    """
    
    inputs, labels = [], []
    
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        
        if 0.1 * i == 0.5:
            continue
            
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

def save_results(save_path, x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize = 12)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else: 
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict Result', fontsize = 12)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else: 
            plt.plot(x[i][0], x[i][1], 'bo')
            
    # plt.show()
    plt.savefig(save_path)
    plt.clf()

def save_lr_curve(save_path, learning_epoch, learning_loss):
    # saving learning curve
    plt.figure()
    plt.title('Learning curve', fontsize=12)
    plt.plot(learning_epoch, learning_loss)
    plt.savefig(save_path)
    plt.clf()