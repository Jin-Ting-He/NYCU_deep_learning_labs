from model import Model
from utils import *

if __name__ == "__main__":
    epochs = 100000
    lr = 0.1
    hidden_unit = 4
    activation = 'sigmoid'
    optimizer = 'adam'

    linear_model = Model(lr = lr, hidden_unit = hidden_unit,
                    activation = activation, optimizer=optimizer)
    
    input, labels = generate_linear()

    learning_epoch, learning_loss = [], []
    # train
    for epoch in range(epochs):
        predict = linear_model.forward(input)
            
        # mse loss
        loss = np.mean((predict - labels) ** 2)
        linear_model.backward(2 * (predict - labels) / len(labels))
        linear_model.optimize()

        if epoch % 500 == 0:
            print(f'Epoch {epoch:<5d}        loss : {loss}')
            learning_epoch.append(epoch)
            learning_loss.append(loss)
    
    # test
    prediction = linear_model.forward(input)
    loss = np.mean((prediction - labels) ** 2)
    
    for i, (gt, pred) in enumerate(zip(labels, prediction)):
        print(f'Iter{i:<3d}  |  Ground Truth:  {gt[0]}  |   Prediction:  {pred[0]:.3f}')
    print(f'Loss : {loss}   Accuracy : {float(np.sum(np.round(prediction) == labels)) * 100 / len(labels)}%')
    prediction = np.round(prediction)

    save_lr_curve("results/linear_sigmoid_lr01_hidden4_sigmoid_adam_curve.png", learning_epoch, learning_loss)

    save_results("results/linear_sigmoid_lr01_hidden4_sigmoid_adam_results.png", input, labels, prediction)