
import netty
import nnfs
from nnfs.datasets import spiral_data
import numpy as np
import matplotlib.pyplot as plt

nnfs.init()


def training(x, y, epochs, optimizer, relu, loss_activation,
             loss_history, accuracy_history, **layers):

    layer1 = layers["layer1"]
    layer2 = layers["layer2"]

    for i in range(epochs):

        layer1.forward(x)
        relu.forward(layer1.output)
        layer2.forward(relu.output)

        loss = loss_activation.forward(layer2.output, y)

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



def loss_acc_plot(loss_history, accuracy_history, title = ""):
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

    plt.tight_layout()
    plt.show()


# ==========================================
# 2. VISUALIZZAZIONE DECISION BOUNDARY
# ==========================================

def plot_decision_boundary(x, y, layer1, relu, layer2):
    # Limiti del piano
    x_min, x_max = x[:, 0].min() - 0.2, x[:, 0].max() + 0.2
    y_min, y_max = x[:, 1].min() - 0.2, x[:, 1].max() + 0.2

    # Creazione di una griglia 2D
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    # Conversione griglia in punti (N, 2)
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # Forward pass su tutti i punti della griglia
    layer1.forward(grid)
    relu.forward(layer1.output)
    layer2.forward(relu.output)

    # Classe predetta per ogni punto
    predictions = np.argmax(layer2.output, axis=1)
    predictions = predictions.reshape(xx.shape)

    # Disegno delle regioni
    plt.figure(figsize=(8, 7))

    plt.contourf(
        xx, yy, predictions,
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap="brg",
        alpha=0.35
    )

    # Confini tra le classi
    plt.contour(
        xx, yy, predictions,
        levels=[0.5, 1.5],
        colors="black",
        linewidths=0.7
    )

    # Punti originali del dataset
    plt.scatter(
        x[:, 0], x[:, 1],
        c=y,
        cmap="brg",
        edgecolors="black",
        linewidths=0.4,
        s=35
    )

    plt.title("Decision Boundary della Rete Neurale")
    plt.xlabel("Feature X1")
    plt.ylabel("Feature X2")
    plt.axis("equal")
    plt.show()


def init_nn():
    x, y = spiral_data(samples=100, classes=3)

    n_input = x.shape[1]
    n_neurons = 512

    layer1 = netty.Layer_Dense(n_input, n_neurons)
    relu = netty.Activation_ReLU()
    layer2 = netty.Layer_Dense(n_neurons, 3)

    loss_activation = netty.Activation_Softmax_Loss_CategoricalCrossentropy()
    return x, y, layer1, layer2, relu, loss_activation

# Inizializzazione rete Adam
x_ada, y_ada, layer1_ada, layer2_ada, relu_ada, loss_activation_ada = init_nn()

# Inizializzazione rete RMSprop
x_rsm, y_rsm, layer1_rsm, layer2_rsm, relu_rsm, loss_activation_rsm = init_nn()

lr = 0.004
d = 0.0001

# Adam
optimizer_ada = netty.Optimizer_AdamW(
    learning_rate= lr,
    decay=d,
    phi = 0.05
)

loss_history_ada = []
accuracy_history_ada = []


# RMSprop
optimizer_rmsprop = netty.Optimizer_Adam(
    learning_rate=lr,
    decay=d,

)

loss_history_rmsprop = []
accuracy_history_rmsprop = []


epochs = 50000


# Training con Adam
training(
    x_ada, y_ada, epochs,
    optimizer_ada, relu_ada, loss_activation_ada,
    loss_history_ada, accuracy_history_ada,
    layer1=layer1_ada, layer2=layer2_ada
)


# Training con RMSprop
training(
    x_rsm, y_rsm, epochs,
    optimizer_rmsprop, relu_rsm, loss_activation_rsm,
    loss_history_rmsprop, accuracy_history_rmsprop,
    layer1=layer1_rsm, layer2=layer2_rsm
)


loss_acc_plot_vs(
    loss_history_ada,
    accuracy_history_ada,
    loss_history_rmsprop,
    accuracy_history_rmsprop,
    "adamW",
    "adam"
)


# plot_decision_boundary(x, y, layer1, relu, layer2)
