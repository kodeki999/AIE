
import netty
import nnfs
from nnfs.datasets import spiral_data
import numpy as np
import matplotlib.pyplot as plt
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os

from pathlib import Path



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
             loss_history, accuracy_history, dropout, **layers):

    layer1 = layers["layer1"]
    layer2 = layers["layer2"]

    for i in range(epochs):

        layer1.forward(x)
        relu.forward(layer1.output)
        dropout.forward(relu.output)
        layer2.forward(dropout.output)

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
        dropout.backward(relu.dinputs)
        layer1.backward(dropout.dinputs)

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
    dropout = netty.Layer_Dropout(0.4)
    optimizer = netty.Optimizer_Adam(
        learning_rate= lr,
        decay=d,
    )

    loss_activation = netty.Activation_Softmax_Loss_CategoricalCrossentropy()
    return layer1, layer2, relu, loss_activation, optimizer, dropout

def megatest():
    # =====================================================
    # 1. CONFIGURAZIONE ESPERIMENTI
    # =====================================================

    n_classes = 3

    N_TRIALS = 200
    BASE_SEED = 42

    OUTPUT_DIR = Path("grid_search_results")
    PLOTS_DIR = OUTPUT_DIR / "plots"

    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


    # =====================================================
    # 2. VALORI DA TESTARE: 10 PER OGNI VARIABILE
    # =====================================================

    param_grid = {

        "lr": np.logspace(-4, -2, 10),

        "d": [
            0.999,
            0.9992,
            0.9994,
            0.9996,
            0.9998,
            0.9999,
            0.99995,
            0.99999,
            0.999995,
            1.0
        ],

        "l2": [
            0,
            1e-6,
            5e-6,
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2
        ],

        "neurons": [
            16, 24, 32, 48, 64,
            96, 128, 256, 384, 512
        ],

        "samples": np.linspace(
            100, 500, 10
        ).round().astype(int),

        "epochs": np.linspace(
            1000, 10000, 10
        ).round().astype(int)
    }


    # =====================================================
    # 3. GENERAZIONE RANDOM SEARCH BILANCIATA
    # =====================================================

    rng = np.random.default_rng(BASE_SEED)

    sampled_params = {}

    for name, values in param_grid.items():

        values = np.asarray(values)

        repetitions = int(np.ceil(N_TRIALS / len(values)))

        expanded = np.tile(values, repetitions)

        expanded = expanded[:N_TRIALS].copy()

        rng.shuffle(expanded)

        sampled_params[name] = expanded


    experiments = []

    for i in range(N_TRIALS):

        config = {
            name: values[i].item()
            for name, values in sampled_params.items()
        }

        experiments.append(config)


    # =====================================================
    # 4. ESECUZIONE DEGLI ESPERIMENTI
    # =====================================================

    results = []

    for run_id, config in enumerate(experiments, start=1):

        lr = config["lr"]
        d = config["d"]
        l2 = config["l2"]
        neurons = config["neurons"]
        samples = config["samples"]
        epochs = config["epochs"]

        print("\n" + "=" * 60)

        print(f"ESPERIMENTO {run_id}/{N_TRIALS}")

        print(f"LR: {lr}")
        print(f"Decay: {d}")
        print(f"L2: {l2}")
        print(f"Neurons: {neurons}")
        print(f"Samples: {samples}")
        print(f"Epochs: {epochs}")

        print("=" * 60)

        try:

            # Seed riproducibile per ogni esperimento
            np.random.seed(BASE_SEED + run_id)

            # ---------------------------------------------
            # GENERAZIONE DATASET
            # ---------------------------------------------

            x, y = spiral_data(
                samples=samples,
                classes=n_classes
            )

            x_tr, y_tr, x_te, y_te = netty.split_dataset(x, y)

            # ---------------------------------------------
            # INIZIALIZZAZIONE RETE
            # ---------------------------------------------

            layer1, layer2, relu, loss_activation, optimizer = init_nn(
                lr,
                d,
                n_classes,
                n_neurons=neurons,
                l2=l2
            )

            loss_history = []
            accuracy_history = []

            # ---------------------------------------------
            # TRAINING
            # ---------------------------------------------

            acc, loss = training(
                x_tr[1],
                y_tr[1],
                epochs,
                optimizer,
                relu,
                loss_activation,
                loss_history,
                accuracy_history,
                layer1=layer1,
                layer2=layer2
            )

            # ---------------------------------------------
            # GRAFICO LOSS E ACCURACY
            # ---------------------------------------------

            loss_plot = loss_acc_plot(
                loss_history,
                accuracy_history,
                title=f"Run {run_id} | acc: {acc} - loss: {loss}"
            )

            loss_plot.savefig(
                PLOTS_DIR / f"acc_loss_run_{run_id}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(loss_plot)

            # ---------------------------------------------
            # DECISION BOUNDARY
            # ---------------------------------------------

            decision_bound_plot = plot_decision_boundary(
                x, y, layer1, relu, layer2
            )

            decision_bound_plot.savefig(
                PLOTS_DIR / f"decision_bound_run_{run_id}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(decision_bound_plot)

            # ---------------------------------------------
            # TEST SET
            # ---------------------------------------------

            acc_test, loss_test = forward_pass(
                x_te,
                y_te,
                relu,
                loss_activation,
                layer1=layer1,
                layer2=layer2
            )

            # ---------------------------------------------
            # SALVATAGGIO RISULTATI
            # ---------------------------------------------

            result = {

                "run": run_id,

                "lr": lr,
                "d": d,
                "l2": l2,
                "neurons": neurons,
                "samples": samples,
                "epochs": epochs,

                "train_accuracy": float(acc),
                "train_loss": float(loss),

                "test_accuracy": float(acc_test),
                "test_loss": float(loss_test),

                "status": "OK"
            }

            results.append(result)

            print(f"Train accuracy: {acc}")
            print(f"Test accuracy: {acc_test}")
            print(f"Test loss: {loss_test}")

        except Exception as error:

            print(f"Errore nell'esperimento {run_id}: {error}")

            results.append({

                **config,

                "run": run_id,
                "train_accuracy": np.nan,
                "train_loss": np.nan,
                "test_accuracy": np.nan,
                "test_loss": np.nan,
                "status": str(error)
            })


        # Salvataggio progressivo: non perdi i risultati
        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "results_progress.csv",
            index=False
        )


    # =====================================================
    # 5. DATAFRAME FINALE
    # =====================================================

    df = pd.DataFrame(results)

    df_valid = df[
        (df["status"] == "OK") &
        (df["test_accuracy"].notna()) &
        (df["test_loss"].notna())
    ].copy()

    df_valid = df_valid.sort_values(
        by="test_accuracy",
        ascending=False
    )

    print("\nTOP 10 CONFIGURAZIONI")

    print(
        df_valid[
            [
                "lr", "d", "l2",
                "neurons", "samples", "epochs",
                "test_accuracy", "test_loss"
            ]
        ].head(10)
    )


    # =====================================================
    # 6. MATRICE DI COVARIANZA
    # =====================================================

    variables = [
        "lr",
        "d",
        "l2",
        "neurons",
        "samples",
        "epochs"
    ]

    metrics = [
        "test_accuracy",
        "test_loss"
    ]

    covariance_data = df_valid[variables + metrics]

    covariance_matrix = covariance_data.cov()

    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(
        covariance_matrix,
        cmap="coolwarm",
        aspect="auto"
    )

    ax.set_xticks(range(len(covariance_matrix.columns)))
    ax.set_yticks(range(len(covariance_matrix.index)))

    ax.set_xticklabels(
        covariance_matrix.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(covariance_matrix.index)

    for i in range(len(covariance_matrix)):
        for j in range(len(covariance_matrix.columns)):

            ax.text(
                j, i,
                f"{covariance_matrix.iloc[i, j]:.3g}",
                ha="center",
                va="center",
                fontsize=8
            )

    fig.colorbar(im, ax=ax, label="Covarianza")

    ax.set_title("Covarianza - Iperparametri, Accuracy e Loss")

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "covariance_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


    # =====================================================
    # 7. CORRELAZIONE NORMALIZZATA
    # =====================================================

    correlation_matrix = covariance_data.corr()

    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))

    ax.set_xticklabels(
        correlation_matrix.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(correlation_matrix.index)

    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix.columns)):

            ax.text(
                j, i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    fig.colorbar(im, ax=ax, label="Correlazione")

    ax.set_title("Correlazione - Iperparametri, Accuracy e Loss")

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


    # =====================================================
    # 8. COVARIANZA DIRETTA CON ACCURACY E LOSS
    # =====================================================

    cov_metrics = covariance_matrix.loc[variables, metrics]

    ax = cov_metrics.plot(
        kind="bar",
        figsize=(12, 6)
    )

    ax.set_title("Covarianza delle variabili rispetto ad Accuracy e Loss")
    ax.set_ylabel("Covarianza")
    ax.set_xlabel("Iperparametri")

    plt.xticks(rotation=30)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "covariance_metrics.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =====================================================
    # 9. ESPORTAZIONE EXCEL CON PANDAS
    # =====================================================

    parameter_table = pd.DataFrame({
        name: values
        for name, values in param_grid.items()
    })

    top10 = df_valid.head(10)

    with pd.ExcelWriter(
        OUTPUT_DIR / "neural_network_results.xlsx",
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="All Experiments",
            index=False
        )

        top10.to_excel(
            writer,
            sheet_name="Top 10",
            index=False
        )

        parameter_table.to_excel(
            writer,
            sheet_name="Parameter Values",
            index=False
        )

        covariance_matrix.to_excel(
            writer,
            sheet_name="Covariance"
        )

        correlation_matrix.to_excel(
            writer,
            sheet_name="Correlation"
        )

        cov_metrics.to_excel(
            writer,
            sheet_name="Metrics Covariance"
        )


    # =====================================================
    # 10. RISULTATO FINALE
    # =====================================================

    print("\n" + "=" * 60)

    print("GRID SEARCH COMPLETATA")

    print(f"Esperimenti eseguiti: {len(df)}")
    print(f"Esperimenti validi: {len(df_valid)}")

    if len(df_valid) > 0:

        best = df_valid.iloc[0]

        print("\nMIGLIORE CONFIGURAZIONE")

        for parameter in variables:
            print(f"{parameter}: {best[parameter]}")

        print(f"Accuracy test: {best['test_accuracy']}")
        print(f"Loss test: {best['test_loss']}")

    print("\nFile Excel:")
    print(OUTPUT_DIR / "neural_network_results.xlsx")

    print("\nGrafici:")
    print(PLOTS_DIR)

    print("=" * 60)


# ___________________________
# impostami dei test automatici per diversi valori delle seguenti variabili: lr, d, l2, nuerons samples (numero di dati fa da 100 a 500 campioni) ed epochs.
# Fammi 10 valori per variabile scrivimi, ti tutte quante conserva i grafici che stampo attualemente, fammi una tabella excell con pandas che posso consultare e fammi un grafico di covarinaza delle variabili rispetto alla accuracye e alla loss

n_classes = 3

x, y = spiral_data(samples=411, classes=n_classes)

x_tr, y_tr, x_te, y_te = netty.split_dataset(x, y)


lr = 0.003
d =  0.999
l2 = 5e-5
neurons = 512

print("=" * 50)
print(f"neuroni: {neurons}\nlambda l2: {l2}\nlearning rate : {lr}\nlr decay: {d}")
print("=" * 50)
print("\n\n")

# Inizializzazione rete Adam
layer1, layer2, relu, loss_activation, optimizer, dropout = init_nn(lr, d, n_classes, n_neurons=neurons, l2 = l2)
loss_history = []
accuracy_history = []
epochs = 4000

acc, loss = training(
    x_tr[1], y_tr[1], epochs,
    optimizer, relu, loss_activation,
    loss_history, accuracy_history, dropout,
    layer1=layer1, layer2=layer2
)
# Grafico Loss e Accuracy
loss_plot = loss_acc_plot(loss_history, accuracy_history, title=f"acc: {acc} - loss: {loss}")
loss_plot.savefig(f"acc_loss_run_.png", dpi=300, bbox_inches="tight")
plt.close(loss_plot)
# Decision Boundary
decision_bound_plot = plot_decision_boundary(x, y, layer1, relu, layer2)
decision_bound_plot.savefig(f"decision_bound_run.png", dpi=300, bbox_inches="tight")
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