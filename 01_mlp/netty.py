import numpy as np

class Layer_Dense:
    def __init__(self, n_input, n_neurons):
        self.weights = 0.01 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    
    def forward(self, inputs):

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss():
    def calculate(self, output, y):
        samples_losses = self.forward(output, y)
        data_loss = np.mean(samples_losses)
        return data_loss



class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
         correct_confidence = y_pred_clipped[
             range(samples),
             y_true
         ]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likehoods = -np.log(correct_confidence)
        return negative_log_likehoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs/samples


class Optimizer_SGD:
    def __init__(self, learning_rate = 1.0, decay = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def post_update_params(self):
        self.iterations += 1

    def update_params(self, layer):
        layer.weights += -  self.current_learning_rate * layer.dweights
        layer.biases   += - self.current_learning_rate * layer.dbiases


class Optimizer_SGD_mom(Optimizer_SGD):
    def __init__(self, learning_rate = 1.0, decay = 0., mom=0.):
        super().__init__(learning_rate, decay)
        self.mom = mom
        self.iterations = 0

    def update_params(self, layer):

        if not hasattr(layer, "weights_mom"):
            layer.weights_mom = np.zeros_like(layer.weights)
            layer.biases_mom = np.zeros_like(layer.biases)

        weights_update = self.mom * layer.weights_mom - self.current_learning_rate * layer.dweights
        layer.weights_mom = weights_update
        biases_update = self.mom * layer.biases_mom  - self.current_learning_rate * layer.dbiases
        layer.biases_mom = biases_update

        layer.weights += weights_update
        layer.biases += biases_update



class Optimizer_AdaGrad(Optimizer_SGD):
    
    def __init__(self, learning_rate=1, decay=0, eps = 1e-7):
        super().__init__(learning_rate, decay)
        self.eps = eps

    def update_params(self, layer):

        if not hasattr(layer, "weights_cache"):
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_cache  = np.zeros_like(layer.biases)


        layer.weights_cache += layer.dweights**2
        layer.biases_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weights_cache) + self.eps)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache) + self.eps)




class Optimizer_RMSprop(Optimizer_SGD):
    
    def __init__(self, learning_rate=1, decay=0, eps = 1e-7, rho = 0.9):
        super().__init__(learning_rate, decay)
        self.eps = eps
        self.rho = rho
    def update_params(self, layer):

        if not hasattr(layer, "weights_cache"):
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_cache  = np.zeros_like(layer.biases)

        layer.weights_cache =  self.rho * layer.weights_cache + (1 - self.rho) * layer.dweights**2
        layer.biases_cache =   self.rho * layer.biases_cache + (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weights_cache) + self.eps)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache) + self.eps)






class Optimizer_Adam(Optimizer_SGD):

    def __init__(self, learning_rate=1, decay=0, eps=1e-7,
                 beta_1=0.9, beta_2=0.999):

        super().__init__(learning_rate, decay)

        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):
                
                
        if not hasattr(layer, "weights_cache"):
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.weights_mom = np.zeros_like(layer.weights)
            layer.biases_cache  = np.zeros_like(layer.biases)
            layer.biases_mom = np.zeros_like(layer.biases)

        layer.weights_mom = self.beta_1 * layer.weights_mom + (1 - self.beta_1) * layer.dweights 
        layer.biases_mom = self.beta_1 * layer.biases_mom + (1 - self.beta_1) * layer.dbiases 

        self.corrected_weights_mom = layer.weights_mom / (1-self.beta_1 ** (self.iterations + 1)) 
        self.corrected_biases_mom = layer.biases_mom / (1-self.beta_1 ** (self.iterations + 1))

        layer.weights_cache = self.beta_2 * layer.weights_cache + (1 - self.beta_2) * layer.dweights**2
        layer.biases_cache = self.beta_2 * layer.biases_cache + (1 - self.beta_2) * layer.dbiases**2
        
        self.corrected_weights_cache = layer.weights_cache / (1-self.beta_2 ** (self.iterations + 1))
        self.corrected_biases_cache = layer.biases_cache  / (1-self.beta_2 ** (self.iterations + 1))

        layer.weights += - self.current_learning_rate * self.corrected_weights_mom / (np.sqrt(self.corrected_weights_cache) + self.eps)
        layer.biases += - self.current_learning_rate *  self.corrected_biases_mom / (np.sqrt( self.corrected_biases_cache) + self.eps)



class Optimizer_AdamW(Optimizer_SGD):

    def __init__(self, learning_rate=1, decay=0, eps=1e-7,
        beta_1=0.9, beta_2=0.999, phi = 0.01):

        super().__init__(learning_rate, decay)
        self.phi = phi
        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    def update_params(self, layer):
                
                
        if not hasattr(layer, "weights_cache"):
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.weights_mom = np.zeros_like(layer.weights)
            layer.biases_cache  = np.zeros_like(layer.biases)
            layer.biases_mom = np.zeros_like(layer.biases)


        layer.weights_mom = self.beta_1 * layer.weights_mom + (1 - self.beta_1) * layer.dweights 
        layer.biases_mom = self.beta_1 * layer.biases_mom + (1 - self.beta_1) * layer.dbiases 

        self.corrected_weights_mom = layer.weights_mom / (1-self.beta_1 ** (self.iterations + 1)) 
        self.corrected_biases_mom = layer.biases_mom / (1-self.beta_1 ** (self.iterations + 1))

        layer.weights_cache = self.beta_2 * layer.weights_cache + (1 - self.beta_2) * layer.dweights**2
        layer.biases_cache = self.beta_2 * layer.biases_cache + (1 - self.beta_2) * layer.dbiases**2
        
        self.corrected_weights_cache = layer.weights_cache / (1-self.beta_2 ** (self.iterations + 1))
        self.corrected_biases_cache = layer.biases_cache  / (1-self.beta_2 ** (self.iterations + 1))

        layer.weights += - self.current_learning_rate * (self.corrected_weights_mom/(np.sqrt(self.corrected_weights_cache) + self.eps)) - self.current_learning_rate * self.phi * layer.weights
        layer.biases += - self.current_learning_rate *  (self.corrected_biases_mom / (np.sqrt( self.corrected_biases_cache) + self.eps))




def accuracy(output, ys):
    pred = np.argmax(output, axis = 1)
    if len(ys.shape) == 2:
        ys = np.argmax(ys, axis = 1)

    return np.mean(pred == ys)