import random as rd
import numpy as np
import matplotlib.pyplot as plt


o1 = np.array([2, 2])
o2 = np.array([1, 1])


def Pcloud(origin: list, numOfP: int, r: float, label: float):

    cloud = []

    for i in range(numOfP):
        v = np.random.randn(len(origin))
        v = (rd.random() * r) * v / np.linalg.norm(v)
        vec = origin + v
        vect = np.append(vec, label)
        cloud.append(vect)


    return cloud



def Pcloud_test(origin: list, numOfP: int, r: float):

    cloud = []

    for i in range(numOfP):
        v = np.random.randn(len(origin))
        v = (rd.random() * r) * v / np.linalg.norm(v)
        vec = origin + v
        cloud.append(vec)


    return cloud


def plot_punti(*punti):

    for punto in punti:
        punto = np.array(punto)
        x = punto[:, 0]
        y = punto[:, 1]

        plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

def plot_punti_retta(w, b, *punti):
    w = np.array(w)

    # Vettore ortogonale a w
    n = np.array([-w[1], w[0]])

    # Parametro della retta
    t = np.linspace(-100, 100, 100)

    p0 = np.array([0, -b / w[1]])
    d = np.array([-w[1], w[0]])

    retta = p0 + t[:, None] * d

    # Disegna i punti
    for punto in punti:
        punto = np.array(punto)

        x = punto[:, 0]
        y = punto[:, 1]

        plt.scatter(x, y)

    # Disegna la retta ortogonale
    plt.plot(retta[:, 0], retta[:, 1])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def training(epoch: int, dataset: list, lr: float):
    w = np.random.random(len(dataset[0])-1)
    b = 1
    for i in range(epoch):
        for vect in dataset:
            vec = vect[0:len(vect)-1]
            label = vect[len(vect)-1]
            dot = np.sign(np.vdot(vec, w) + b)

            error = label - dot
            b = b + lr * error
            w = w + vec * error * lr

    return w, b

def classify( weights, bais ,points,):
   
    return [np.sign(np.vdot(i, weights) + bais) for i in points]


clusterA = Pcloud(o1, 500, 0.5, -1)
clusterB = Pcloud(o2, 500, 0.5, 1)


dataset = clusterA + clusterB



w, b = training(50, dataset, 0.05)


test_points_a = Pcloud_test(o1, 10, 0.5)
test_points_b = Pcloud_test(o2, 10, 0.5)

test_a = np.array(classify(w, b ,test_points_a))
test_b = np.array(classify(w, b ,test_points_b))

print(test_a)
print(test_b)

# plot_punti(clusterA, clusterB)

plot_punti_retta(w, b,clusterA, clusterB)

# con un learning rate più alto e meno epoche converge male, per poco stabili
# learning rate basso e tante epoche

