import netty
import nnfs
from nnfs.datasets import sine_data
import numpy as np
import matplotlib.pyplot as plt

nnfs.init()

def create_data(samples=1000):
    X = np.random.uniform(-10, 10, (samples, 1))

    y = np.sin(X * 0.5) * 4 + np.arctan(X * 9) * 8 

    return X, y



def init_nn():
    x, y = create_data()

    n_input = x.shape[1]
    n_neurons = 64

    layer1 = netty.Layer_Dense(n_input, n_neurons)
    relu1 = netty.Activation_ReLU()
    layer2 = netty.Layer_Dense(n_neurons, n_neurons)
    relu2 = netty.Activation_ReLU()
    layer3 = netty.Layer_Dense(n_neurons, 1)


    lin_activation = netty.Activation_Linear()
    loss_fun = netty.Loss_MeanSquaredError()
    optimazier = netty.Optimizer_Adam(learning_rate=0.0001)
    accuracy_precision = np.std(y) / 250

    return x, y, layer1, layer2, layer3, relu1, relu2, lin_activation, loss_fun, optimazier, accuracy_precision



def training(x, y, epochs, optimizer, relu1,relu2, lin_act, loss_fun, accuracy_precision,
             loss_history, accuracy_history, **layers):

    layer1 = layers["layer1"]
    layer2 = layers["layer2"]
    layer3 = layers["layer3"]

    for i in range(epochs):

        layer1.forward(x)
        relu1.forward(layer1.output)
        layer2.forward(relu1.output)
        relu2.forward(layer2.output)
        layer3.forward(relu2.output)
        lin_act.forward(layer3.output)
        data_loss = loss_fun.calculate(lin_act.output, y)

        regularization_loss = loss_fun.regularization_loss(layer1) + loss_fun.regularization_loss(layer2) + loss_fun.regularization_loss(layer3) 
            
        loss = data_loss + regularization_loss

        predictions = lin_act.output
        accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)



        loss_history.append(loss)
        accuracy_history.append(accuracy)

        if i % 100 == 0:
            print(f"Epoch: {i} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

        loss_fun.backward(lin_act.output, y)
        lin_act.backward(loss_fun.dinputs)
        layer3.backward(lin_act.dinputs)
        relu2.backward(layer3.dinputs)
        layer2.backward(relu2.dinputs)
        relu1.backward(layer2.dinputs)
        layer1.backward(relu1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.update_params(layer3)
        optimizer.post_update_params()

    return layer1, relu1, layer2, relu2, layer3, lin_act

def plot_test(dense1, activation1, dense2, activation2, dense3, activation3):
    x_test, y_test = sine_data()
    dense1.forward(x_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    plt.plot(x_test, y_test)
    plt.plot(x_test, activation3.output)
    plt.show()

def plot_train(dense1, activation1, dense2, activation2, dense3, activation3, x, y):
    
    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    sort_idx = np.argsort(x[:, 0])

    plt.scatter(x[:, 0], y[:, 0], label="Real", s=15)
    plt.plot(
        x[sort_idx, 0],
        activation3.output[sort_idx, 0],
        label="Predicted",
        color="red"
    )

    plt.legend()
    plt.grid()
    plt.show()




x, y, layer1, layer2, layer3, relu1, relu2, lin_activation, loss_fun, optimazier, accuracy_precision = init_nn()


loss_history = []
accuracy_history = []

layer1, relu1, layer2, relu2, layer3, lin_act = training(x, y, 20000, optimazier, relu1, relu2, lin_activation, loss_fun, accuracy_precision, loss_history, accuracy_history, layer1 = layer1, layer2 = layer2, layer3 = layer3)


plot_train(layer1, relu1, layer2, relu2, layer3, lin_act, x, y)