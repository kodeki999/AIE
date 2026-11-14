
import netty
import nnfs
from nnfs.datasets import spiral_data
import numpy as np
import matplotlib.pyplot as plt
import math
nnfs.init()


def forward_pass(x, y,  relu, loss_activation, **layers):

    layer1 = layers["layer1"]
    layer2 = layers["layer2"]

    layer1.forward(x)
    relu.forward(layer1.output)
    layer2.forward(relu.output)
    # Calculate loss from output of activation2 so softmax activation
    data_loss = loss_activation.forward(layer2.output, y)
    # Calculate regularization penalty
    regularization_loss = loss_activation.regularization_loss(layer1) + loss_activation.regularization_loss(layer2)
    # Calculate overall loss
    loss = data_loss + regularization_loss
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy, loss




def training(x, y, epochs, optimizer, relu, loss_activation,
             loss_history, accuracy_history, **layers):

    layer1 = layers["layer1"]
    layer2 = layers["layer2"]

    for i in range(epochs):

        layer1.forward(x)
        relu.forward(layer1.output)
        layer2.forward(relu.output)

        # Calculate loss from output of activation2 so softmax activation
        data_loss = loss_activation.forward(layer2.output, y)
        # Calculate regularization penalty
        regularization_loss = loss_activation.regularization_loss(layer1) + loss_activation.regularization_loss(layer2)
        # Calculate overall loss
        loss = data_loss + regularization_loss

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)

        loss_history.append(loss)
        accuracy_history.append(accuracy)

        if i % 100 == 0:
            print(f"Epoch: {i} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

        loss_activation.backward(loss_activation.output, y)
        layer2.backward(loss_activation.dinputs)
        relu.backward(layer2.dinputs)
        layer1.backward(relu.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.post_update_params()

    return accuracy_history[-1], loss_history[-1]
# ==========================================
# 1. GRAFICO LOSS E ACCURACY
# ==========================================






def loss_acc_plot_vs(loss_adam, accuracy_adam,
                  loss_rmsprop, accuracy_rmsprop, title1 = "", title2 = ""):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confronto Loss
    axes[0].plot(loss_adam, color="red", label=title1, linewidth=1.5)
    axes[0].plot(loss_rmsprop, color="orange", label=title2, linewidth=1.5)
    axes[0].set_title("Confronto Loss")
    axes[0].set_xlabel("Epoche")
    axes[0].set_ylabel("Categorical Crossentropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Confronto Accuracy
    axes[1].plot(accuracy_adam, color="blue", label=title1, linewidth=1.5)
    axes[1].plot(accuracy_rmsprop, color="green", label=title2, linewidth=1.5)
    axes[1].set_title("Confronto Accuracy")
    axes[1].set_xlabel("Epoche")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



def loss_acc_plot(loss_history, accuracy_history, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(loss_history, color="red", linewidth=1.5)
    axes[0].set_title(f"Andamento della Loss {title}")
    axes[0].set_xlabel("Epoche")
    axes[0].set_ylabel("Categorical Crossentropy")
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(accuracy_history, color="blue", linewidth=1.5)
    axes[1].set_title(f"Andamento dell'Accuracy {title}")
    axes[1].set_xlabel("Epoche")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    return fig


# ==========================================
# 2. VISUALIZZAZIONE DECISION BOUNDARY
# ==========================================

def plot_decision_boundary(x, y, layer1, relu, layer2):

    x_min, x_max = x[:, 0].min() - 0.2, x[:, 0].max() + 0.2
    y_min, y_max = x[:, 1].min() - 0.2, x[:, 1].max() + 0.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    layer1.forward(grid)
    relu.forward(layer1.output)
    layer2.forward(relu.output)

    predictions = np.argmax(layer2.output, axis=1)
    predictions = predictions.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.contourf(
        xx, yy, predictions,
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap="brg",
        alpha=0.35
    )

    ax.contour(
        xx, yy, predictions,
        levels=[0.5, 1.5],
        colors="black",
        linewidths=0.7
    )

    ax.scatter(
        x[:, 0], x[:, 1],
        c=y,
        cmap="brg",
        edgecolors="black",
        linewidths=0.4,
        s=35
    )

    ax.set_title("Decision Boundary della Rete Neurale")
    ax.set_xlabel("Feature X1")
    ax.set_ylabel("Feature X2")
    ax.axis("equal")

    return fig





def init_nn(lr, d, n_classes, n_neurons = 512, l2 = 0.1):

    
    lr = 0.04
    d = 0.00001


    n_input = x.shape[1]
    

    layer1 = netty.Layer_Dense(n_input, n_neurons, weight_regularizer_l2=l2,
                                                   bias_regularizer_l2  =l2)
    relu = netty.Activation_ReLU()
    layer2 = netty.Layer_Dense(n_neurons, n_classes,
                               weight_regularizer_l2=l2,
                               bias_regularizer_l2  =l2)

    optimizer = netty.Optimizer_Adam(
        learning_rate= lr,
        decay=d,
    )

    loss_activation = netty.Activation_Softmax_Loss_CategoricalCrossentropy()
    return layer1, layer2, relu, loss_activation, optimizer

n_classes = 3

x, y = spiral_data(samples=100, classes=n_classes)

x_tr, y_tr, x_te, y_te = netty.split_dataset(x, y)





for i,e in enumerate(x_tr):

    lr = 1
    d =  0.00001

    # l2 = (-i * 0.5 + 1) + 0.01
    # l2 = math.exp(-i**2)
    # l2 = math.log(i+1)
    l2 = 0.001
    neurons = 512
 
    print("=" * 50)
    print(f"neuroni: {neurons}\nlambda l2: {l2}\nlearning rate : {lr}\nlr decay: {d}")
    print("=" * 50)
    print("\n\n")
    # Inizializzazione rete Adam
    layer1, layer2, relu, loss_activation, optimizer = init_nn(lr, d, n_classes, n_neurons=neurons, l2 = l2)

    loss_history = []
    accuracy_history = []

    epochs = 10000

    
    acc, loss = training(
        e, y_tr[i], epochs,
        optimizer, relu, loss_activation,
        loss_history, accuracy_history,
        layer1=layer1, layer2=layer2
    )



    # Grafico Loss e Accuracy
    loss_plot = loss_acc_plot(loss_history, accuracy_history, title=f"acc: {acc} - loss: {loss}")
    loss_plot.savefig(f"acc{acc}_loss{loss}_run_{i}.png", dpi=300, bbox_inches="tight")
    plt.close(loss_plot)


    # Decision Boundary
    decision_bound_plot = plot_decision_boundary(x, y, layer1, relu, layer2)
    decision_bound_plot.savefig(f"decision_bound_run{i}.png", dpi=300, bbox_inches="tight")
    plt.close(decision_bound_plot)
    print("Grafico salvato! Programma in esecuzione...")

    acc_test, loss_test = forward_pass(
        x_te, y_te, relu, loss_activation,
        layer1=layer1, layer2=layer2
        )

    print("\n\n")
    print("=" * 50)
    print(f"accuratezza su test set: {acc_test}\nloss su test set: {loss_test}")
    print("=" * 50)
    print("\n\n")

    
    